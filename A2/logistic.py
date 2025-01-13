from utils import Array, sigmoid
import numpy as np
from numpy import log


def logistic_predict(weights: Array, data: Array) -> Array:
    """Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    # YOUR CODE BEGINS HERE

    y_lst = []
    N, M = data.shape
    temp = np.ones((N, 1))
    data = np.append(data, temp, axis=1)
    # z = w^Tx = x^Tw for each x in X
    for i in range(N):
        x = data[i]
        lst = []
        for k in range(len(weights[0])):
            row_sum = 0
            for j in range(len(x)):
                row_sum += x[j] * weights[j][0]
            lst.append(row_sum)

        z_i = np.array(lst)
        y_lst.append(sigmoid(z_i))
    y = np.array(y_lst)
    y.reshape(-1, 1)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets: Array, y: Array) -> tuple[float, float]:
    """Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    # YOUR CODE BEGINS HERE
    ce = -np.mean(targets * np.log(y + 1e-15) + (1 - targets) * np.log(1 - y + 1e-15))

    predictions = (y >= 0.5).astype(int)
    frac_correct = np.mean(predictions == targets)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(
        weights: Array, data: Array, targets: Array, hyperparameters: dict
) -> tuple[float, Array, Array]:
    """Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    # YOUR CODE BEGINS HERE

    f, frac = evaluate(targets, y)
    y_t = []  # y - t
    for i in range(len(targets)):
        y_t.append(y[i][0] - targets[i][0])
    df_lst = []
    N,M = data.shape
    temp = np.ones((N, 1))
    data_new = np.append(data, temp, axis=1)
    for i in range(len(data_new[0])):  # calculate frac{1}{N} (y-t)X
        row_sum = 0
        for j in range(len(y_t)):
            row_sum += y_t[j] * data_new[j][i]
        df_lst.append(row_sum / len(y_t))
    df = np.array(df_lst).reshape(-1, 1)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
