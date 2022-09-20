import math
import torch
import numpy as np
from torch import nn
from torch.nn import BatchNorm1d
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F

def to_one_hot(inp,num_classes):
    device = inp.device
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.long().unsqueeze(1).data.cpu(), 1)
    return Variable(y_onehot.to(device), requires_grad=False)

class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class GBN(torch.nn.Module):
    """
        Ghost Batch Normalization
        https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)

class EmbeddingGenerator(torch.nn.Module):
    """
        Classical embeddings generator
        adopted from https://github.com/dreamquark-ai/tabnet/
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim=[]):
        """ This is an embedding module for an enite set of features
        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embdeding dimension will be used for all categorical features
        """
        super(EmbeddingGenerator, self).__init__()
        if cat_dims == [] or cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return

        # heuristic
        if (len(cat_emb_dim) == 0):
            # use heuristic
            cat_emb_dim = [min(600, round(1.6 * n_cats ** .56)) for n_cats in cat_dims]

        self.skip_embedding = False
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim]*len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = """ cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims))

        self.embeddings = torch.nn.ModuleList()

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, int(emb_dim)))
        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embdeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x
        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(self.embeddings[cat_feat_counter](x[:, feat_init_idx].long()))
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings

def get_lambda(alpha=0.5, dist='alpha', n=1):
    '''Return lambda'''
    if alpha > 0.:
        if dist == 'alpha':
            lam1 = np.random.beta(alpha, alpha, size=n)
            lam2 = np.random.beta(alpha, alpha, size=n)
        elif dist == 'uniform':
            lam1 = np.random.uniform(0.0, alpha, size=n)
            lam2 = np.random.uniform(0.0, alpha, size=n)
    else:
        lam1 = 1.0
        lam2 = 1.0
    return lam1, lam2


def mixup_process(out, target_reweighted, lam1, lam2):
    indices1 = np.random.permutation(out.size(0))
    indices2 = np.random.permutation(out.size(0))
    out = out*lam1 + out[indices1]*lam2 + out[indices2]*(1-lam1-lam2)
    target_shuffled_onehot1 = target_reweighted[indices1]
    target_shuffled_onehot2 = target_reweighted[indices2]
    target_reweighted = target_reweighted * lam1 + target_shuffled_onehot1 * lam2 + target_shuffled_onehot2 * (1-lam1-lam2)
    return out, target_reweighted, indices1, indices2


def mixup_class(out, labels, lam1, lam2):
    yk = torch.unique(labels)
    if (len(lam1) == len(yk)): # - class wise lamda
        yk = list(zip(lam1, lam2, yk))

    new_zs = []
    idxs1 = []
    idxs2 = []
    perms1 = []
    perms2 = []
    lam1_ = lam1
    lam2_ = lam2
    for y in yk: # TODO: how to parallelize this
        if type(y) == tuple:
            lam1_, lam2_, y = y
        idx = labels == y
        if len(lam1) == len(out):
            lam1_ = lam1[idx].reshape(-1, 1)  # - boradcast along features
            lam2_ = lam2[idx].reshape(-1, 1)
        idx = torch.arange(idx.size(0))[idx]
        perm1 = torch.randperm(idx.size(0))
        perm2 = torch.randperm(idx.size(0))
        idx_perm1 = idx[perm1]
        idx_perm2 = idx[perm2]
        zns = lam1_ * out[idx] + lam2_ * out[idx_perm1] + (1-lam1_-lam2_) * out[idx_perm2]
        new_zs.append(zns)
        perms1.append(idx_perm1)
        perms2.append(idx_perm2)
        idxs1.append(idx)
        idxs2.append(idx)

    return torch.cat(new_zs, axis=0), torch.cat(idxs1, axis=0), torch.cat(idxs2, axis=0), torch.cat(perms1, axis=0), torch.cat(perms2, axis=0)


def mixup_process_label_free(out, lam1, lam2):
    indices1 = np.random.permutation(out.size(0))
    indices1 = torch.Tensor(indices1).long()
    indices2 = np.random.permutation(out.size(0))
    indices2 = torch.Tensor(indices2).long()
    out = out * lam1 + out[indices1] * lam2 + out[indices2] * (1 - lam1 - lam2)
    return out, indices1, indices2


def mixup(out, indices1, indices2, lam1, lam2):
    if len(out.shape) == 1:  # - 1d vector
        lam1 = lam1.squeeze()  # - avoid broadcast below
        lam2 = lam2.squeeze()
    out_idxs = torch.arange(len(indices1)) % len(out)
    out = out[out_idxs] * lam1 + out[indices1] * lam2 + out[indices2] * (1 - lam1 - lam2)
    return out

def mixup_full_indices(out, lambda_indices1, lambda_indices2, one_minus_lambda_indices1, one_minus_lambda_indices2, lam1, lam2):
    if len(out.shape) == 1:  # - 1d vector
        lam1 = lam1.squeeze()  # - avoid broadcast below
        lam2 = lam2.squeeze()
    return lam1 * out[lambda_indices1] + lam2 * out[one_minus_lambda_indices1] + (1 - lam1 - lam2) * out[one_minus_lambda_indices2]

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)

