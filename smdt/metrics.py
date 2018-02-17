
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
import numpy as np
import pandas as pd

def get_classifier_metrics(models, y_test, y_pred, y_score, positive_label):
    """

    """
    fpr = []
    tpr = []
    roc_auc = []
    hamming = []
    accuracy = []
    f1 = []
    f_beta1 = []
    f_beta2 = []
    matt_corr = []
    jac_sim = []
    prec = []
    rec = []
    z1l = []

    for i in range(len(y_pred)):
        if models[i] == 'Linear SVC' or models[i] == 'SGD Classifier':
            fpr_, tpr_, _ = roc_curve(y_test, y_score[i], pos_label=positive_label)
        else:
            fpr_, tpr_, _ = roc_curve(y_test, y_score[i][:, 1], pos_label=positive_label)
        fpr.append(fpr_)
        tpr.append(tpr_)
        roc_auc.append(auc(fpr_, tpr_))
        f1.append(metrics.f1_score(y_test, y_pred[i], pos_label=positive_label))
        prec.append(metrics.precision_score(y_test, y_pred[i], pos_label=positive_label))
        rec.append(metrics.recall_score(y_test, y_pred[i], pos_label=positive_label))

        accuracy.append(metrics.accuracy_score(y_test, y_pred[i]))
        hamming.append(metrics.hamming_loss(y_test, y_pred[i]))
        matt_corr.append(metrics.matthews_corrcoef(y_test, y_pred[i]))
        jac_sim.append(metrics.jaccard_similarity_score(y_test, y_pred[i]))
        z1l.append(metrics.zero_one_loss(y_test, y_pred[i]))

    a = pd.DataFrame.from_dict({'Models': models, 'Accuracy': accuracy, 'Area under ROC': roc_auc, 'Precision': prec,
                                'Recall': rec, 'F1 Score': f1, 'Hamming Loss': hamming,
                                'Matthews Correlation': matt_corr, 'Jaccard Similarity': jac_sim,
                                'Zero One Loss': z1l, }, orient='index').T

    fig = plt.figure(figsize=(15, 5))
    plt.subplot(121)
    lw = 2
    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i], lw=lw, label='%s (area = %0.2f)' % (models[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.subplot(122)
    for i in range(len(fpr)):
        if models[i] == 'Linear SVC' or models[i] == 'SGD Classifier':
            precision, recall, _ = precision_recall_curve(y_test, y_score[i], pos_label=positive_label)
        else:
            precision, recall, _ = precision_recall_curve(y_test, y_score[i][:, 1], pos_label=positive_label)
        plt.step(recall, precision, label=models[i], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")

    cnf_matrix = []
    # Compute confusion matrix
    for i in range(len(y_pred)):
        cnf_matrix.append(confusion_matrix(y_test, y_pred[i]))

    np.set_printoptions(precision=2)

    plt.figure(figsize=(12, 10))
    for i in range(len(models)):
        plt.subplot(100 + len(models) * 10 + 1 + i)
        plot_confusion_matrix(cnf_matrix[i], classes=y_test.unique(),
                              title=models[i])

    plt.show()

    return a


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')