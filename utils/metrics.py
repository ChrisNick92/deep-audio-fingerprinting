from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np


def acc_score(y_true, y_pred):
    return round(accuracy_score(y_true, y_pred), 4)


def f1(y_true, y_pred):
    return round(f1_score(y_true, y_pred, labels=list(set(y_true)), zero_division=0, average='macro'), 4)


def recall(y_true, y_pred):
    return round(recall_score(y_true, y_pred, labels=list(set(y_true)), zero_division=0, average='macro'), 4)


def precision(y_true, y_pred):
    return round(precision_score(y_true, y_pred, labels=list(set(y_true)), zero_division=0, average='macro'), 4)


def top_1_hit_rate(y_true, y_pred, offset_true, offset_pred, tolerance=0.5):
    on_time = np.where(np.abs(offset_pred - offset_true) <= tolerance, 1, 0)
    matches = np.where(y_true == y_pred, 1, 0)
    misses = np.where(np.logical_or(y_true != y_pred, on_time != 1), 1, 0)
    return round(np.sum(on_time * matches) / (np.sum(on_time * matches) + np.sum(misses)), 4)


def summary_metrics(y_true, y_pred, offset_true, offset_pred, tolerance=0.5):
    top_1 = top_1_hit_rate(y_true, y_pred, offset_true, offset_pred)
    res = f'Accuracy: {100 * acc_score(y_true, y_pred):.2f} %\n' + \
        f'F1: {100*f1(y_true, y_pred):.2f} %\n'+ \
        f'Recall : {100* recall(y_true, y_pred):.2f} %\n'+ \
        f'Precision: {100* precision(y_true, y_pred):.2f} %\n' + \
        f'Top 1 hit rate: {100 * top_1:.2f} %\n'

    print(res)

    return top_1