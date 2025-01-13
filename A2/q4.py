# -*- coding: utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from numpy import e

from sklearn.model_selection import train_test_split

#####################################################################
# NOTE: Here LAM is the hard-coded value of lambda for LRLS
# NOTE: Feel free to play with lambda as well if you wish
#####################################################################
LAM = 1e-5

# For tpye contracts

Array = np.ndarray


# helper function
def l2(A: Array, B: Array) -> Array:
    """
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    """
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm + B_norm - 2 * A.dot(B.transpose())
    return dist


# to implement
def LRLS(
        test_datum: Array,
        x_train: Array,
        y_train: Array,
        tau: float,
        lam: float = LAM,
) -> Array:
    """
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    """
    #####################################################################
    # TODO: Implement LRLS function in Q4(b).
    #####################################################################
    # YOUR CODE BEGINS HERE
    N, D = x_train.shape
    d_sum = 0
    for i in range(N):
        d_sum += e ** (- (np.linalg.norm(test_datum - x_train[i].reshape(-1, 1)) ** 2) / (2 * (tau ** 2)))
    a_i = []
    for i in range(N):
        num = e ** (- (np.linalg.norm(test_datum - x_train[i].reshape(-1, 1)) ** 2) / (2 * (tau ** 2)))
        if d_sum == 0:
            a_i.append(0)
        else:
            a_i.append(num / d_sum)
    A = np.diag(np.array(a_i))
    lam_matrix = np.diag(np.array([lam for i in range(D)]))

    temp = np.matmul(np.matmul(np.transpose(x_train), A), x_train) + lam_matrix

    w_star = np.matmul(np.matmul(np.matmul(np.linalg.inv(temp), np.transpose(x_train)), A), y_train)

    y_hat = np.matmul(np.transpose(test_datum), w_star)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y_hat


def run_validation(
        x: Array, y: Array, taus: Array, val_frac: float
) -> tuple[list[float], list[float]]:
    """
    Input: x is the N x d design matrix
           y is the 1-dimensional vector of size N
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    """
    #####################################################################
    # TODO: Complete the rest of the code for Q4(c).
    #####################################################################
    train_losses = []
    validation_losses = []
    # YOUR CODE BEGINS HERE

    X_train, X_valid, y_train, y_valid = train_test_split(x, y, train_size=1 - val_frac)
    N_valid, D = X_valid.shape
    # for valid set
    for t in taus:
        loss_sum = 0
        for i in range(N_valid):
            x_i = X_valid[i]
            pred_y = LRLS(x_i.reshape(-1, 1), X_train, y_train, t, LAM)
            loss_sum += (pred_y - y_valid[i]) ** 2
        validation_losses.append(loss_sum / N_valid)

    N_train, D = X_train.shape
    # for train set
    for t in taus:
        loss_sum = 0
        for i in range(N_train):
            x_i = X_train[i]
            pred_y = LRLS(x_i.reshape(-1, 1), X_train, y_train, t, LAM)
            loss_sum += (pred_y - y_train[i]) ** 2
        train_losses.append(loss_sum / N_train)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return train_losses, validation_losses


if __name__ == "__main__":
    # feel free to change this number depending on resource usage
    import os

    NUM_TAUS = os.environ.get("NUM_TAUS", 200)  # the graph in writeup uses 50 instead of 200
    #####################################################################
    #                       DO NOT MODIFY CODE BELOW                   #
    #####################################################################
    from sklearn.datasets import fetch_california_housing

    np.random.seed(0)
    # load boston housing prices dataset
    housing = fetch_california_housing()
    n_samples = 500
    x = housing["data"][:n_samples]
    N = x.shape[0]
    # add constant one feature - no bias needed
    x = np.concatenate((np.ones((N, 1)), x), axis=1)
    d = x.shape[1]
    y = housing["target"][:N]
    taus = np.logspace(1, 3, NUM_TAUS)
    train_losses, test_losses = run_validation(x, y, taus, val_frac=0.3)
    plt.semilogx(taus, train_losses, label="Training Loss")
    plt.semilogx(taus, test_losses, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.xlabel("Tau values (log scale)")
    plt.ylabel("Average squared error loss")
    plt.title("Training and Validation Loss w.r.t Tau")
    plt.savefig("q4.png")