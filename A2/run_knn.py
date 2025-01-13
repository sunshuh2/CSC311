from utils import Array, load_train, load_valid, load_test
import matplotlib.pyplot as plt
import numpy as np


# helper function
def l2_distance(A: Array, B: Array) -> Array:
    """
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the 2-norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||_2
    """
    A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
    B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
    dist = np.sqrt(A_norm + B_norm - 2 * A.dot(B.transpose()))
    return dist


def knn(k: int, train_data: Array, train_labels: Array, valid_data: Array) -> Array:
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data, train_data)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def accuracy(A, B):
    diff = 0
    for i in range(len(A)):
        if A[i] == B[i]:
            diff += 1
    return diff / len(A)


def run_knn(k_vals: list[int]) -> None:
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    k_accuracy = []

    for k in k_vals:
        predict_valid = knn(k, train_inputs, train_targets, valid_inputs)
        k_accuracy.append(accuracy(predict_valid, valid_targets))

    plt.plot(k_vals, k_accuracy)
    plt.title("k vs accuracy")

    # 3.1 b

    chosen_k = k_vals[k_accuracy.index(max(k_accuracy))]
    print(chosen_k)
    accuracy_val = max(k_accuracy)
    predict_test = knn(chosen_k, train_inputs, train_targets, test_inputs)
    accuracy_test = accuracy(predict_test, test_targets)
    print("for k*:", "accuracy on validation:", accuracy_val, "accuracy on test:", accuracy_test)

    # k+2,k-2
    predict_val = knn(chosen_k + 2, train_inputs, train_targets, valid_inputs)
    accuracy_val_p2 = accuracy(predict_val, valid_targets)
    predict_test = knn(chosen_k + 2, train_inputs, train_targets, test_inputs)
    accuracy_test_p2 = accuracy(predict_test, test_targets)
    print("for k*+2:", "accuracy on validation:", accuracy_val_p2, "accuracy on test:", accuracy_test_p2)
    if chosen_k > 2:
        predict_val = knn(chosen_k - 2, train_inputs, train_targets, valid_inputs)
        accuracy_val_m2 = accuracy(predict_val, valid_targets)
        predict_test = knn(chosen_k - 2, train_inputs, train_targets, test_inputs)
        accuracy_test_m2 = accuracy(predict_test, test_targets)
    print("for k*-2:", "accuracy on validation:", accuracy_val_m2, "accuracy on test:", accuracy_test_m2)

    # the validation and test accuracies increases as k star increases and decreases when k star decreases

    #####################################################################
    # TODO:                                                             #
    # Implement a function that runs kNN for different values of k,     #
    # plots the classification rate on the validation set, and etc.     #
    # as required for Q3.1 (a) and (b).                                 #
    #####################################################################

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    # This function needs to save a plot (with the appropriate Information)
    # named "knn.png" to be used for automated grading. Do NOT change it!
    plt.savefig("knn.png")


if __name__ == "__main__":
    #####################################################################
    #                       DO NOT MODIFY CODE BELOW                   #
    #####################################################################
    run_knn([1, 3, 5, 7, 9])
