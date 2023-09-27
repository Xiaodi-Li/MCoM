import os.path

import numpy as np
import torch
from keras.datasets import mnist
from torch.utils.data import Dataset
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

METHODS = ['', 'supervised', 'semisupervised', 'pseudolabeling']

class Security(Dataset):
    """ Implements keras MNIST torch.utils.data.dataset


        Args:
            train (bool): flag to determine if train set or test set should be loaded
            labeled_ratio (float): fraction of train set to use as labeled
            method (str): valid methods are in `METHODS`.
                'supervised': getitem return x_labeled, y_label
                'semisupervised': getitem will return x_labeled, x_unlabeled, y_label
                'pseudolabeling': getitem will return x_labeled, x_unlabeled, y_label, y_pseudolabel
            random_seed (int): change random_seed to obtain different splits otherwise the fixed random_seed will be used

    """

    # Need this function to decode embeddings before feeding it into the model.
    def decode_embedding_float(self, array, vector_size):
        def _to_bytes(x):
            if x is None or len(x) == 0: return list(b'\x00' * vector_size)
            vs = x[::-1].tobytes()[::-1]
            return list(vs[:vector_size])

        big_endian_bytes = [_to_bytes(x) for x in array]
        array = np.array(big_endian_bytes, dtype='<i1') * 1.0
        return array

    def transform(self, features):
        columns = features.columns
        for col in sorted(columns):
            if col == 'hasVendorScore' or col == 'isAbusingVendor':
                features[col] = features[col].astype(int)
            if col == 'productCpi' or col == 'auditCoupangAdjustedPrice' \
                    or col == 'auditBenchmarkPrice' or col == 'coupangAdjustedPrice' \
                    or col == 'benchmarkPrice' or col == 'rocket' \
                    or col == 'taxonomyId' or col == 'category6RepurchaseRate' or col == 'icategoryName' \
                    or col == 'priceVariation1M' or col == 'prices1M' \
                    or col == 'icategoryTag' or col == 'cqiCategoryString' \
                    or col == 'title' or col == 'icategoryTag' or col == 'icategoryCode' \
                    or col == 'reviewScore' or col == 'brand' \
                    or col == 'categoryCode' or col=='priceTrendCurr1m' or col=='priceTrendCurr3m' \
                    or col=='categoryCodeLeaf' or col =='pricePercentile' or col=='floorPriceBucket0' \
                    or col == 'winnerVendorId_henry' or col == 'winnerVendorId_hash' or col == 'winnerƒVendorId_dnn':
                features = features.drop(col, axis=1)
                continue
            if features[col].dtypes == 'object':
                if col == 'productBertEmbedding':
                    emb_arr = self.decode_embedding_float(features['productBertEmbedding'], 32)
                    col_pbe = ['pbe' + str(i) for i in range(len(features['productBertEmbedding'].to_list()[0]))]
                    pbe_df = pd.DataFrame(emb_arr, columns=col_pbe, dtype=float)
                    features = features.drop(col, axis=1)
                    # features = pd.concat([features, pbe_df], axis=1) ##not sure why valid data behaves weirdly. will return this later.
                elif col == 'winnerVendorId_vid':
                    features.rename(columns={col: "winnerVendorId_hash"})
                elif col == 'title_emb':
                    emb_arr = features['title_emb'].to_list()
                    col_tes = ['tes' + str(i) for i in range(len(features['title_emb_softmax'].to_list()[0]))]
                    tes_df = pd.DataFrame(emb_arr, columns=col_tes, dtype=float)
                    features = features.drop(col, axis=1)
                elif col == 'title_emb_softmax':
                    # emb_arr = features['title_emb_softmax'].to_list()
                    # col_tes = ['tes' + str(i) for i in range(len(features['title_emb_softmax'].to_list()[0]))]
                    # tes_df = pd.DataFrame(emb_arr, columns=col_tes, dtype=float)
                    features = features.drop(col, axis=1)
                    # features = pd.concat([features, tes_df], axis=1)
                elif col == 'category6RepurchaseRate' or col == 'category3RepurchaseRate' or col == 'productRepurchaseRate':
                    convert = []
                    for s in features[col]:
                        if s:
                            convert.append(float(s.split('_')[2]))
                        else:
                            convert.append(hash(s))
                    features[col] = convert
                else:
                    col_hash = [hash(s) for s in features[col]]
                    features[col] = col_hash
        # features = pd.DataFrame(np.hstack([features, pbe_df]), columns=list(features.columns) + col_pbe, dtype=float)
        # features = pd.DataFrame(np.hstack([features, tes_df]), columns=list(features.columns) + col_tes, dtype=float)
        return features

    def __init__(self,
                 train: bool,
                 labeled_ratio: float=0.0,
                 method: str='supervised',
                 random_seed: int=None,
                 data_dir: str='./dataset/DC/data/qupk/relevance_project/q3_food_tail_project/training_data/20230925/parquet_data'):
        super().__init__()
        print('Dataloader __getitem__ mode: {}'.format(method))
        assert method.lower() in METHODS , 'Method argument is invalid {}, must be in'.format(METHODS)
        train_df = pd.read_parquet(os.path.join(data_dir, 'train')).drop(['doc_id'], axis=1)
        valid_df = pd.read_parquet(os.path.join(data_dir, 'valid')).drop(['doc_id'], axis=1)
        test_df = pd.read_parquet(os.path.join(data_dir, 'test')).drop(['doc_id'], axis=1)
        train_df.dropna(axis=1, inplace=True)
        valid_df.dropna(axis=1, inplace=True)
        test_df.dropna(axis=1, inplace=True)
        # train_df.drop([name] for name in train_df if train_df[name].dtype == 'string')
        columns_to_redefine = [name for name in train_df.columns if train_df[name].dtype == 'object']  # 'object' is the dtype for strings in pandas
        for name in columns_to_redefine:
            train_df[name] = pd.factorize(train_df[name])[0]
        columns_to_redefine = [name for name in valid_df.columns if valid_df[name].dtype == 'object']  # 'object' is the dtype for strings in pandas
        for name in columns_to_redefine:
            valid_df[name] = pd.factorize(valid_df[name])[0]
        columns_to_redefine = [name for name in test_df.columns if test_df[name].dtype == 'object']  # 'object' is the dtype for strings in pandas
        for name in columns_to_redefine:
            test_df[name] = pd.factorize(test_df[name])[0]

        data_train = train_df
        data_test = test_df
        print(data_train)
        print(f'feature size = {len(list(data_train.columns)) - 1}')

        data_train_y = data_train['label'].to_numpy().astype(np.int)
        data_train_x = data_train.drop('label', axis=1).to_numpy().astype(np.float32)

        data_test_y = data_test['label'].to_numpy().astype(np.int)
        data_test_x = data_test.drop('label', axis=1).to_numpy().astype(np.float32)

        print(len(data_train_y))
        print(len(data_test_y))
        num = np.unique(data_train_y)
        print('y unique number: {}'.format(len(num)))

        # data_csv.rename(columns={"CAN ID": "CAN_ID"}, inplace=True)
        # # Convert the Dataset Hex Value to Decimal
        # data_csv[["CAN_ID", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]] = data_csv[
        #     ["CAN_ID", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7"]
        # ].apply(lambda x: x.astype(str).map(lambda x: int(x, base=16)))
        # # #############################################################################
        # # Delete the values in Label Column consists of Zero
        # data_csv.drop(data_csv.index[data_csv["Label"] == "0"], inplace=True)
        # data_csv.Label = pd.factorize(data_csv.Label)[0]

        # data_x = data_df.to_numpy().astype(np.float32)
        # print(f'feature size = {data_x.shape}')
        # data_eval_x = data_eval.to_numpy().astype(np.float32)
        # data_eval_y = Y_eval.astype(np.int)

        # data augmentation triplet
        idx0 = [i for i, x in enumerate(data_train_y) if x == 0]
        idx = [i for i, x in enumerate(data_train_y) if x == 1]
        idx_test = [i for i, x in enumerate(data_test_y) if x == 1]
        print(len(idx0))
        print(len(idx))
        print(len(idx_test))
        # idx = random.sample(idx, 60)
        aug_orig = data_train_x[idx]
        aug_new_x = []
        aug_new_y = []
        lmd1 = np.random.uniform(low=0.1, high=0.5)
        lmd2 = np.random.uniform(low=0.1, high=0.5)
        for i in range(0, len(aug_orig) - 2):
            for j in range(i + 1, len(aug_orig) - 1, 100000):
                for k in range(j + 1, len(aug_orig), 10000):
                    aug_new_x.append(lmd1 * aug_orig[i] + lmd2 * aug_orig[j] + (1 - lmd1 - lmd2) * aug_orig[k])
                    aug_new_y.append(1)
        data_train_x = np.concatenate((data_train_x, aug_new_x), axis=0)
        data_train_y = np.append(data_train_y, aug_new_y)
        print(len(data_train_y))

        # random data set
        rand_idx = random.sample(range(0, len(data_train_y)), len(data_train_y))
        data_train_x = data_train_x[rand_idx]
        data_train_y = data_train_y[rand_idx]

        rand_idx = random.sample(range(0, len(data_test_y)), len(data_test_y))
        data_test_x = data_test_x[rand_idx]
        data_test_y = data_test_y[rand_idx]
        ###convert numpy to pandas dataframe, pd.to_parquet('path')

        data_train = np.hstack((data_train_x, np.expand_dims(data_train_y,1)))
        data_train_pandas = pd.DataFrame(data_train)
        data_train_pandas.columns = train_df.columns
        data_train_pandas.to_parquet('data_train_dc.parquet', engine='pyarrow')

        data_test = np.hstack((data_test_x, np.expand_dims(data_test_y,1)))
        data_test_pandas = pd.DataFrame(data_test, columns=test_df.columns)
        data_test_pandas.to_parquet('data_test_dc.parquet', engine='pyarrow')

        # data_eval['label'] = Y_eval
        valid_df.to_parquet('data_valid_dc.parquet', engine='pyarrow')

        print('save done!')



        if (train):
            self.data = data_train_x
            self.targets = data_train_y
            # (self.data, self.targets), (_, _) = mnist.load_data()
        else:
            self.data = data_test_x
            self.targets = data_test_y
            # (_, _), (self.data, self.targets) = mnist.load_data()

        # self.data = self.data / 255.0
        self.train = train
        # self.data = np.reshape(self.data, (len(self.targets), -1)).astype(np.float32)
        self.method = method.lower()
        idx = np.arange(len(self.targets))
        self.labeled_idx = idx
        self.unlabeled_idx = idx
        self.idx = idx
        if (labeled_ratio > 0):
            if random_seed is not None:
                idx = np.random.RandomState(seed=random_seed).permutation(len(self.targets))
            else:
                idx = np.random.permutation(len(self.targets))
            self.idx = idx
            if labeled_ratio <= 1.0:
                ns = labeled_ratio * len(self.idx)
            else:
                ns = labeled_ratio
            ns = int(ns)
            labeled_idx = self.idx[:ns]
            unlabeled_idx = self.idx[ns:]
            self.labeled_idx = labeled_idx
            self.unlabeled_idx = unlabeled_idx

        self._pseudo_labels = list()
        self._pseudo_labels_weights = list()

    def get_pseudo_labels(self):
        return self._pseudo_labels

    def set_pseudo_labels(self, pseudo_labels):
        self._pseudo_labels = pseudo_labels

    def set_pseudo_labels_weights(self, pseudo_labels_weights):
        self._pseudo_labels_weights = pseudo_labels_weights

    def __len__(self):
        if (self.method == 'pseudolabeling'):
            return len(self.idx)
        return len(self.labeled_idx)

    def _semisupervised__getitem__(self, idx):
        idx = self.labeled_idx[idx]
        uidx = np.random.randint(0, len(self.unlabeled_idx))
        uidx = self.unlabeled_idx[uidx]
        img, target = self.data[idx], int(self.targets[idx])
        uimg = self.data[uidx]
        if len(self._pseudo_labels):
            utarget = self._pseudo_labels[uidx]
            uweight = self._pseudo_labels_weights[uidx]
            return img, uimg, target, utarget, uweight
        return img, uimg, target

    def _normal__getitem__(self, idx):
        idx = self.labeled_idx[idx]
        img, target = self.data[idx], int(self.targets[idx])
        return img, target

    def _pseudolabeling__getitem__(self, idx):
        idx = self.idx[idx]
        img, target = self.data[idx], int(self.targets[idx])
        labeled_mask = np.array([False], dtype=np.bool)
        if idx in self.labeled_idx:
            labeled_mask[0] = True
        idx = np.asarray([idx])
        return img, target, labeled_mask, idx

    def __getitem__(self, idx):
        if self.method == 'semisupervised' and self.train:
            return self._semisupervised__getitem__(idx)
        if self.method == 'pseudolabeling' and self.train:
            return self._pseudolabeling__getitem__(idx)
        else:
            return self._normal__getitem__(idx)
