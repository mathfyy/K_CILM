"""
functions used to calculate the metrics for multi-label classification
cmap=mAP, emap=MiAP
"""
import numpy as np
import pdb

from sklearn import metrics
import warnings

warnings.filterwarnings("ignore")


def compute_tagbias_acc_s_m(preds, targs, thr=0.5):
    if np.size(preds) == 0:
        return 0.0
    auc = np.zeros((preds.shape[1]))
    mcc = np.zeros((preds.shape[1]))
    uf1 = np.zeros((preds.shape[1]))
    uar = np.zeros((preds.shape[1]))

    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        scores = scores[targets != -1]
        targets = targets[targets != -1]
        if len(targets) == 0:
            auc[k] = -1
            mcc[k] = -1
            uf1[k] = -1
            uar[k] = -1
            continue
        fpr, tpr, thresholds = metrics.roc_curve(targets, scores)
        auc[k] = metrics.auc(fpr, tpr)
        mcc[k] = metrics.matthews_corrcoef(targets, (scores > thr).astype(np.float32))
        uf1[k] = metrics.f1_score(targets, (scores > thr).astype(np.float32), average='macro', zero_division=1)
        uar[k] = metrics.recall_score(targets, (scores > thr).astype(np.float32), average='macro', zero_division=1)

    auc = auc[auc != -1]
    mcc = mcc[mcc != -1]
    uf1 = uf1[uf1 != -1]
    uar = uar[uar != -1]

    y_pred = (preds > thr).astype(np.float32)
    # acc1 = np.sum(targs == y_pred) / np.sum(targs != -1)
    # acc_P1 = np.sum((targs == y_pred) * (targs == 1)) / np.sum(targs == 1)
    # acc_N1 = np.sum((targs == y_pred) * (targs == 0)) / np.sum(targs == 0)
    # 每个类单独算平均
    acc = np.zeros((preds.shape[1]))
    acc_P = np.zeros((preds.shape[1]))
    acc_N = np.zeros((preds.shape[1]))
    for k in range(preds.shape[1]):
        scores = y_pred[:, k]
        targets = targs[:, k]
        scores = scores[targets != -1]
        targets = targets[targets != -1]
        if len(targets) == 0:
            acc[k] = -1
            acc_P[k] = -1
            acc_N[k] = -1
            continue
        acc[k] = np.sum(targets == scores) / np.sum(targets != -1)
        if np.sum(targets == 1) == 0:
            acc_P[k] = 1.0
        else:
            acc_P[k] = (np.sum((targets == scores) * (targets == 1))) / (np.sum(targets == 1))

        if np.sum(targets == 0) == 0:
            acc_P[k] = 1.0
        else:
            acc_N[k] = (np.sum((targets == scores) * (targets == 0))) / (np.sum(targets == 0))

    acc = acc[acc != -1]
    acc_P = acc_P[acc_P != -1]
    acc_N = acc_N[acc_N != -1]

    return 100 * np.nanmean(acc), 100 * np.nanmean(acc_P), 100 * np.nanmean(acc_N), 100 * np.nanmean(auc), np.nanmean(
        mcc), 100 * np.nanmean(uf1), 100 * np.nanmean(uar)


def average_precision(output, target):
    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))
    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    if total == 0 and precision_at_i_ == 0:
        precision_at_i = 1.0
    elif total == 0 and precision_at_i_ != 0:
        precision_at_i = 0.0
    else:
        precision_at_i = precision_at_i_ / total

    return precision_at_i


def calculate_metrics(preds, targets, thre=0.5):
    # epsilon = 1e-8
    prediction = (preds > thre).astype(np.float32)
    tp_c = np.float64((((prediction + targets) == 2) * (targets != -1))).sum(axis=0)
    fp_c = np.float64((((prediction - targets) == 1) * (targets != -1))).sum(axis=0)
    fn_c = np.float64((((prediction - targets) == -1) * (targets != -1))).sum(axis=0)
    tn_c = np.float64((((prediction + targets) == 0) * (targets != -1))).sum(axis=0)

    precision_c = np.zeros(tp_c.shape)
    recall_c = np.zeros(tp_c.shape)
    f1_c = np.zeros(tp_c.shape)

    for i in range(len(tp_c)):
        # precision_c[i] = np.float64(np.float64(tp_c[i] + epsilon) / np.float64(tp_c[i] + fp_c[i] + epsilon)) * 100.0
        if np.float64(tp_c[i] + fp_c[i]) == 0 and np.float64(tp_c[i]) == 0:
            precision_c[i] = 100.0
        elif np.float64(tp_c[i] + fp_c[i]) == 0 and np.float64(tp_c[i]) != 0:
            precision_c[i] = 0.0
        else:
            precision_c[i] = np.float64(np.float64(tp_c[i]) / np.float64(tp_c[i] + fp_c[i])) * 100.0

        # recall_c[i] = np.float64(np.float64(tp_c[i] + epsilon) / np.float64(tp_c[i] + fn_c[i] + epsilon)) * 100.0
        if np.float64(tp_c[i] + fn_c[i]) == 0 and np.float64(tp_c[i]) == 0:
            recall_c[i] = 100.0
        elif np.float64(tp_c[i] + fn_c[i]) == 0 and np.float64(tp_c[i]) != 0:
            recall_c[i] = 0.0
        else:
            recall_c[i] = np.float64(np.float64(tp_c[i]) / np.float64(tp_c[i] + fn_c[i])) * 100.0

        # f1_c[i] = (2 * precision_c[i] * recall_c[i] + epsilon) / (precision_c[i] + recall_c[i] + epsilon)
        if precision_c[i] + recall_c[i] == 0 and precision_c[i] * recall_c[i] == 0:
            f1_c[i] = 100.0
        elif precision_c[i] + recall_c[i] == 0 and precision_c[i] * recall_c[i] != 0:
            f1_c[i] = 0.0
        else:
            f1_c[i] = (2 * precision_c[i] * recall_c[i]) / (precision_c[i] + recall_c[i])

    mean_p_c = sum(precision_c) / len(precision_c)
    mean_r_c = sum(recall_c) / len(recall_c)
    mean_f_c = sum(f1_c) / len(f1_c)

    # precision_o = (tp_c.sum() + epsilon) / ((tp_c + fp_c).sum() + epsilon) * 100.0
    if (tp_c + fp_c).sum() == 0 and tp_c.sum() == 0:
        precision_o = 100.0
    elif (tp_c + fp_c).sum() == 0 and tp_c.sum() != 0:
        precision_o = 0.0
    else:
        precision_o = tp_c.sum() / (tp_c + fp_c).sum() * 100.0

    # recall_o = (tp_c.sum() + epsilon) / ((tp_c + fn_c).sum() + epsilon) * 100.0
    if (tp_c + fn_c).sum() == 0 and tp_c.sum() == 0:
        recall_o = 100.0
    elif (tp_c + fn_c).sum() == 0 and tp_c.sum() != 0:
        recall_o = 0.0
    else:
        recall_o = tp_c.sum() / (tp_c + fn_c).sum() * 100.0

    # f1_o = (2 * precision_o * recall_o + epsilon) / (precision_o + recall_o + epsilon)
    if precision_o + recall_o == 0 and precision_o * recall_o == 0:
        f1_o = 100.0
    elif precision_o + recall_o == 0 and precision_o * recall_o != 0:
        f1_o = 0.0
    else:
        f1_o = (2 * precision_o * recall_o) / (precision_o + recall_o)

    if np.size(preds) == 0:
        mAP = 0.0
    else:
        ap = np.zeros((preds.shape[1]))
        # compute average precision for each class
        for k in range(preds.shape[1]):
            # sort scores
            scores = preds[:, k]
            targs = targets[:, k]
            # scores = scores[targs != -1]
            # targs = targs[targs != -1]
            # if len(targs) == 0:
            #     ap[k] = -1
            #     continue
            # compute average precision
            ap[k] = average_precision(scores, targs)
            # if ap[k] == 0:
            #     judge = (scores > 0.5).astype(np.float32)
            #     judge = judge[targs != -1]
            #     judgel = targs[targs != -1]
            #     if len(judgel) == 0:
            #         ap[k] = -1
            #     else:
            #         if np.sum(judge == judgel) == len(judgel):
            #             ap[k] = 1
        # ap = ap[ap != -1]
        mAP = 100 * np.nanmean(ap)
    return mAP, precision_o, recall_o, f1_o, mean_p_c, mean_r_c, mean_f_c
