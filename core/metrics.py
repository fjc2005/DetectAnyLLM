import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, matthews_corrcoef, balanced_accuracy_score

def AUROC(original_discrepancy_list, rewritten_discrepancy_list):
    labels = [0] * len(original_discrepancy_list) + [1] * len(rewritten_discrepancy_list)
    preds = original_discrepancy_list + rewritten_discrepancy_list
    fpr, tpr, _ = roc_curve(labels, preds)
    auroc = auc(fpr, tpr)

    return fpr.tolist(), tpr.tolist(), float(auroc)

def AUPR(original_discrepancy_list, rewritten_discrepancy_list):
    labels = [0] * len(original_discrepancy_list) + [1] * len(rewritten_discrepancy_list)
    preds = original_discrepancy_list + rewritten_discrepancy_list
    precision, recall, _ = precision_recall_curve(labels, preds)
    aupr = auc(recall, precision)

    return precision.tolist(), recall.tolist(), float(aupr)


def MCC(original_discrepancy_list, rewritten_discrepancy_list, threshold=0.5):
    """
    计算马修斯相关系数（Matthews Correlation Coefficient）
    :param threshold: 用于将连续预测值转换为二分类标签的阈值（默认0.5）
    """
    labels = [0] * len(original_discrepancy_list) + [1] * len(rewritten_discrepancy_list)
    preds = original_discrepancy_list + rewritten_discrepancy_list
    # 将连续预测值转换为二分类标签（>threshold为正类1，否则负类0）
    pred_labels = [1 if p > threshold else 0 for p in preds]
    return float(matthews_corrcoef(labels, pred_labels))

def Balanced_Accuracy(original_discrepancy_list, rewritten_discrepancy_list, threshold=0.5):
    """
    计算平衡准确率（Balanced Accuracy）
    :param threshold: 用于将连续预测值转换为二分类标签的阈值（默认0.5）
    """
    labels = [0] * len(original_discrepancy_list) + [1] * len(rewritten_discrepancy_list)
    preds = original_discrepancy_list + rewritten_discrepancy_list
    # 将连续预测值转换为二分类标签（>threshold为正类1，否则负类0）
    pred_labels = [1 if p > threshold else 0 for p in preds]
    return float(balanced_accuracy_score(labels, pred_labels))

def TPR_at_FPR5(original_discrepancy_list, rewritten_discrepancy_list):
    labels = [0] * len(original_discrepancy_list) + [1] * len(rewritten_discrepancy_list)
    preds = original_discrepancy_list + rewritten_discrepancy_list
    fpr, tpr, _ = roc_curve(labels, preds)
    
    # 寻找满足FPR≤5%的索引
    valid_indices = np.where(fpr <= 0.05)[0]
    
    if not valid_indices.size:
        return 0.0  # 理论上不会发生，因为至少有一个点（FPR=0）
    
    i_max = valid_indices[-1]
    
    if i_max == len(fpr) - 1:
        return float(tpr[i_max])
    else:
        # 线性插值
        fpr_low = fpr[i_max]
        fpr_high = fpr[i_max + 1]
        tpr_low = tpr[i_max]
        tpr_high = tpr[i_max + 1]
        
        # 计算插值比例
        interpolate_ratio = (0.05 - fpr_low) / (fpr_high - fpr_low)
        tpr_at_5 = tpr_low + interpolate_ratio * (tpr_high - tpr_low)
        return float(tpr_at_5)