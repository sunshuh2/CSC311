import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)


def knn_impute_by_user(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    return acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("./data").toarray()
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    a = sparse_matrix
    b = []
    for i in range(1774):
        count = 0
        for j in range(524):
            if a[j, i] == 1.0 or a[j,i] == 0.0:
                count += 1
        if count < 10:
            b.append(count)
    print(len(b))
    c = []
    for i in range(524):
        count = 0
        for j in range(1774):
            if a[i,j] == 1.0 or a[i,j] == 0.0:
                count += 1
        if count < 10:
            c.append(count)
    print(len(c))
    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_val = [1, 6, 11, 16, 21, 26]

    # a
    acc_by_user = []
    for k in k_val:
        acc_by_user.append(knn_impute_by_user(sparse_matrix, val_data, k))
    plt.plot(k_val, acc_by_user)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.show()

    # b
    k_star = k_val[acc_by_user.index(max(acc_by_user))]
    print("k*:", k_star)
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_star)
    print("Test Accuracy with k*:", test_acc)

    # c
    acc_by_item = []
    for k in k_val:
        acc_by_item.append(knn_impute_by_item(sparse_matrix, val_data, k))
    plt.plot(k_val, acc_by_item)
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.show()

    # d
    k_star = k_val[acc_by_item.index(max(acc_by_item))]
    print("k*:", k_star)
    test_acc = knn_impute_by_item(sparse_matrix, test_data, k_star)
    print("Test Accuracy with k*:", test_acc)


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
