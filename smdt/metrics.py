
from sklearn.metrics import f1_score, recall_score, make_scorer, classification_report, roc_curve, precision_recall_curve, precision_score, recall_score, accuracy_score, matthews_corrcoef, jaccard_similarity_score, zero_one_loss, auc, roc_auc_score
from sklearn.feature_selection import f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np


def plot_roc(y_test, y_score, target_label, model_name):
    fpr_, tpr_, _ = roc_curve(y_test, y_score, pos_label=target_label)
    plt.plot(fpr_, tpr_, lw=2, label='%s (area = %0.2f)' % (model_name, auc(fpr_,tpr_)))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    return

def plot_prc(y_test, y_score, target_label, model_name):
    precision, recall, _ = precision_recall_curve(y_test, y_score, pos_label=target_label)
    plt.step(recall, precision, label=model_name, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    return


def get_ClassificationMetrics(model_name, y_test, y_pred, y_score, target_label):
    # Metrics
    fpr_, tpr_, _ = roc_curve(y_test, y_score, pos_label=target_label)
    f1 = f1_score(y_test, y_pred, pos_label=target_label)
    prec = precision_score(y_test, y_pred, pos_label=target_label)
    rec = recall_score(y_test, y_pred, pos_label=target_label)
    accuracy = accuracy_score(y_test, y_pred)
    matt_corr = matthews_corrcoef(y_test, y_pred)
    jac_sim = jaccard_similarity_score(y_test, y_pred)
    z1l = zero_one_loss(y_test, y_pred)
    roc_auc = auc(fpr_, tpr_)

    metric = pd.DataFrame.from_dict(
        {'Model': model_name, 'Accuracy': accuracy, 'Area under ROC': roc_auc, 'Precision': prec,
         'Recall': rec, 'F1 Score': f1, 'Matthews Correlation': matt_corr, 'Jaccard Similarity': jac_sim,
         'Zero One Loss': z1l}, orient='index').T
    return metric
