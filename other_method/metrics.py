import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score

def AUROC(neg_list, pos_list):
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = neg_list + pos_list
    fpr, tpr, _ = roc_curve(labels, preds)
    auroc = auc(fpr, tpr)

    return fpr.tolist(), tpr.tolist(), float(auroc)

def AUPR(neg_list, pos_list):
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = neg_list + pos_list
    precision, recall, _ = precision_recall_curve(labels, preds)
    aupr = auc(recall, precision)

    return precision.tolist(), recall.tolist(), float(aupr)


def MCC(neg_list, pos_list, threshold=0.5):
    """
    Calculate Matthews Correlation Coefficient
    :param threshold: Threshold for converting continuous predictions to binary labels (default 0.5)
    """
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = neg_list + pos_list
    # Convert continuous predictions to binary labels (>threshold is positive class 1, otherwise negative class 0)
    pred_labels = [1 if p > threshold else 0 for p in preds]
    return float(matthews_corrcoef(labels, pred_labels))

def Balanced_Accuracy(neg_list, pos_list, threshold=0.5):
    """
    Calculate Balanced Accuracy
    :param threshold: Threshold for converting continuous predictions to binary labels (default 0.5)
    """
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = neg_list + pos_list
    # Convert continuous predictions to binary labels (>threshold is positive class 1, otherwise negative class 0)
    pred_labels = [1 if p > threshold else 0 for p in preds]
    return float(balanced_accuracy_score(labels, pred_labels))

def TPR_at_FPR5(neg_list, pos_list):
    labels = [0] * len(neg_list) + [1] * len(pos_list)
    preds = neg_list + pos_list
    fpr, tpr, _ = roc_curve(labels, preds)
    
    # Find indices where FPR <= 5%
    valid_indices = np.where(fpr <= 0.05)[0]
    
    if not valid_indices.size:
        return 0.0  # This should not happen in theory, because there is at least one point (FPR=0)
    
    i_max = valid_indices[-1]
    
    if i_max == len(fpr) - 1:
        return float(tpr[i_max])
    else:
        # Linear interpolation
        fpr_low = fpr[i_max]
        fpr_high = fpr[i_max + 1]
        tpr_low = tpr[i_max]
        tpr_high = tpr[i_max + 1]
        
        # Calculate interpolation ratio
        interpolate_ratio = (0.05 - fpr_low) / (fpr_high - fpr_low)
        tpr_at_5 = tpr_low + interpolate_ratio * (tpr_high - tpr_low)
        return float(tpr_at_5)