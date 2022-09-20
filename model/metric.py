import torch
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_curve, auc, precision_recall_curve, roc_auc_score


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def precision(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred[target == 1.0] == target[target == 1.0]).item()
    return correct / torch.sum(target)

def precision_value(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        precision = precision_score(target, pred)
    return precision

def recall_value(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        recall = recall_score(target, pred)
    return recall

def tpr_tnr(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
        tpr = float(tp) / (tp + fn)
        tnr = float(tn) / (fp + tn)
    return tpr, tnr

def f1_score_value(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        f1_binary = f1_score(target, pred)
        f1_micro = f1_score(target, pred, average='micro')
        f1_macro = f1_score(target, pred, average='macro')
        f1_weighted = f1_score(target, pred, average='weighted')
    return f1_binary, f1_micro, f1_macro, f1_weighted

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def top_1_acc(output, target):
    return top_k_acc(output, target)

def top_5_acc(output, target):
    return top_5_acc(output, target)

def roc_auc(output, target):
    '''
    calculate receiver operating characteristic
    must be done after each epoch once outputs/targets have been accumulated
    '''
    fpr, tpr, thresholds = roc_curve(target, output[:, -1])
    area = auc(fpr, tpr)
    return area

#def roc_auc_score(output, target, average='micro'):
#    return roc_auc_score(output, target, average=average)

def pr_auc(output, target):
    '''
    calculate preciison recall auc
    must be done after each epoch once outputs/targets have been accumulated
    '''

    precision, recall, _ = precision_recall_curve(target, output[:, -1])
    area = auc(recall, precision)
    return area
