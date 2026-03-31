import numpy as np
import matplotlib.pyplot as plt
# DO NOT import any other libraries.


def confusion_matrix_metrics(conf_mat):
    """
    Compute evaluation metrics from a confusion matrix.

    Parameters
    ----------
    conf_mat : np.ndarray of shape (K, K)
        Confusion matrix where conf_mat[i, j] is the number of
        samples with true label i predicted as label j.

    Returns
    -------
    metrics : np.ndarray of shape (K, 4)
        Each row corresponds to one class.
        Columns are:
        [accuracy, recall, precision, f1_score]
    """

    # TODO:

    conf_mat = np.array(conf_mat)
    K = conf_mat.shape[0]
    total = conf_mat.sum()
    metrics = np.zeros((K, 4))

    for i in range(K):
        tp = conf_mat[i, i]
        fp = conf_mat[:, i].sum() - tp  
        fn = conf_mat[i, :].sum() - tp   
        tn = total - tp - fp - fn

        tp = conf_mat[i, i]
        fp = conf_mat[:, i].sum() - tp  
        fn = conf_mat[i, :].sum() - tp  
        tn = total - tp - fp - fn

        if total > 0:
            accuracy = (tp + tn) / total
        else:
            accuracy = 0.0

        if (tp + fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if (tp + fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if (precision + recall) > 0:
            f1_score = (2 * precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        metrics[i] = [accuracy, recall, precision, f1_score]


    return metrics




def plot_roc_curve(y_true, y_score, num_thresholds=100):
    """
    Plot ROC curve from ground-truth labels and predicted probabilities.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth binary labels (0 or 1)

    y_score : array-like of shape (n_samples,)
        Predicted probabilities for the input

    num_thresholds : int
        Number of thresholds to evaluate (default=100)

    Returns
    -------
    fpr : np.ndarray
        False Positive Rates

    tpr : np.ndarray
        True Positive Rates
    """


    tpr_list = []
    fpr_list = []

    # TODO:
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    thresholds = np.linspace(1.0, 0.0, num_thresholds)

    tpr_list = []
    fpr_list = []

    P = y_true.sum()
    N = len(y_true) - P

    for i in thresholds:
        y_pred = (y_score >= i)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        if P > 0:
            tpr = tp / P
        else:
            tpr = 0.0

        if N > 0:
            fpr = fp / N
        else:
            fpr = 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    red_thresholds = [0.5, 0.8]
    red_points = []
    for i in red_thresholds:
        y_pred = (y_score >= i)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        if P > 0:
            tpr_pts = tp / P
        else:
            tpr_pts = 0.0

        if N > 0:
            fpr_pts = fp / N
        else:
            fpr_pts = 0.0
        red_points.append((fpr_pts, tpr_pts, i))



    # Plot ROC curve
    plt.figure()
    plt.plot(fpr_list, tpr_list, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return np.array(fpr_list), np.array(tpr_list)


# Task 3

y_true_balanced = np.array([1]*500 + [0]*500)
y_score_balanced = np.random.rand(1000)
y_true_imbalanced = np.array([1]*50 + [0]*950)
y_score_imbalanced = np.random.rand(1000)

plot_roc_curve(y_true_balanced, y_score_balanced)
plot_roc_curve(y_true_imbalanced, y_score_imbalanced)