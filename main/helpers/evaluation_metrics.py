from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix as cm,
    classification_report as cr,
    matthews_corrcoef as mcc,
)
import numpy as np
from typing import AnyStr


def classification_report(
        true: np.ndarray,
        pred: np.ndarray
) -> AnyStr:
    """
    Method to return slightly modified classification_report than sklearn.

    :param
        true: True classes value
        pred: Predicted classes value

    :return:
        Structured string with classification_report
    """
    report = cr(true, pred)
    report += f'\n         auc                           {roc_auc_score(true, pred).round(2)}'
    report += f'\n         mcc                           {mcc(true, pred).round(2)} \n'
    return report


def confusion_matrix(
        true: np.ndarray,
        pred: np.ndarray,
        title: AnyStr = "Confusion Matrix",
) -> None:
    """
    Method to plot confusion matrix.

    :param
        true: True classes value
        pred: Predicted classes value

    :return:
        None
    """
    from matplotlib import pyplot as plt
    conf = cm(true, pred)
    plt.clf()
    plt.imshow(conf, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative', 'Positive']
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN', 'FP'], ['FN', 'TP']]

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(conf[i][j]))
    plt.show()
