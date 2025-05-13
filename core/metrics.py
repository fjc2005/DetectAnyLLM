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