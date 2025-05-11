from sklearn.metrics import roc_curve, precision_recall_curve, auc

def AUROC(original_discrepancy_list, rewritten_discrepancy_list):
    labels = [0] * len(original_discrepancy_list) + [1] * len(rewritten_discrepancy_list)
    preds = original_discrepancy_list.extend(rewritten_discrepancy_list)
    fpr, tpr, _ = roc_curve(labels, preds)
    auroc = auc(fpr, tpr)

    return fpr.tolist(), tpr.tolist(), float(auroc)

def AUPR(original_discrepancy_list, rewritten_discrepancy_list):
    labels = [0] * len(original_discrepancy_list) + [1] * len(rewritten_discrepancy_list)
    preds = original_discrepancy_list.extend(rewritten_discrepancy_list)
    precision, recall, _ = precision_recall_curve(labels, preds)
    aupr = auc(recall, precision)

    return precision.tolist(), recall.tolist(), float(aupr)

