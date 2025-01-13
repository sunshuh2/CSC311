"""
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
NOTE: Do not modify or add any more import statements.
"""

import data
import numpy as np
import scipy.special  # might be useful!

# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    """
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    """
    means = np.zeros((10, 64))
    class_count = np.zeros((10, 1))
    N = len(train_labels)
    # Compute means
    for i in range(N):
        y = int(train_labels[i])
        means[y, :] += train_data[i]
        class_count[y, 0] += 1
    return means / class_count


def compute_sigma_mles(train_data, train_labels):
    """
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    """
    covariances = np.zeros((10, 64, 64))
    N = len(train_labels)
    class_count = np.zeros((10, 1))
    mean_mle = compute_mean_mles(train_data, train_labels)
    temp = np.identity(64) * 0.01
    stabilizer = np.repeat(temp[np.newaxis, :, :], 10, axis=0)

    # Compute covariances
    for i in range(N):
        y = int(train_labels[i])
        class_count[y, 0] += 1
        temp = np.matmul((train_data[i] - mean_mle[y]).reshape((64, 1)), (train_data[i] - mean_mle[y]).reshape(1, 64))
        covariances[y, :, :] += temp
    return covariances / class_count.reshape(10, 1, 1) + stabilizer


def generative_likelihood(digits, means, covariances):
    """
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    """
    N = len(digits)
    log_like = np.zeros((N, 10))
    d = 64
    for c in range(10):
        inverse = np.linalg.inv(covariances[c])
        determinate = np.linalg.det(covariances[c])
        temp2 = np.log(((2 * np.pi) ** (-d / 2)) * (determinate ** (-1 / 2)))
        for i in range(N):
            temp = digits[i] - means[c]
            exponential = -0.5 * np.matmul(np.matmul(temp.T, inverse), temp)
            log_like[i, c] = temp2 + exponential
    return log_like


def conditional_likelihood(digits, means, covariances):
    """
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    """
    N = len(digits)
    gen_log_like = generative_likelihood(digits, means, covariances)
    con_log_like = np.full((N, 10), np.log(1 / 10)) + gen_log_like
    sum_class = con_log_like[:, 0]
    for i in range(1, 10):
        sum_class = np.logaddexp(sum_class, con_log_like[:, i])
    for i in range(N):
        con_log_like[i, :] -= sum_class[i]
    return con_log_like


def avg_conditional_likelihood(digits, labels, means, covariances):
    """
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    N = len(labels)
    correct_sum = 0
    for i in range(N):
        y = int(labels[i])
        correct_sum += cond_likelihood[i, y]
    # Compute as described above and return
    return correct_sum / N


def classify_data(digits, means, covariances):
    """
    Classify new points by taking the most likely posterior class
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    N = len(digits)
    classifications = np.zeros(N)
    for i in range(N):
        max_likelihood = np.max(cond_likelihood[i])
        classifications[i] = np.where(cond_likelihood[i] == max_likelihood)[0]
    # Compute and return the most likely class
    return classifications


def accuracy(classification, labels):
    """ Helper function to compute accuracy """
    count = 0
    N = len(labels)
    for i in range(N):
        if classification[i] == int(labels[i]):
            count += 1
    return count / N


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data("data")

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation: Parts (a) - (c)
    print("average conditional log-likelihood on train dataset is:",
          avg_conditional_likelihood(train_data, train_labels, means, covariances))
    print("average conditional log-likelihood on test dataset is:",
          avg_conditional_likelihood(test_data, test_labels, means, covariances))
    # print train, test log-likelihoods and accuracies

    result_train = classify_data(train_data, means, covariances)
    result_test = classify_data(test_data, means, covariances)

    print("accuracy for train dataset is:", accuracy(result_train, train_labels))
    print("accuracy for test dataset is:", accuracy(result_test, test_labels))

    #  covariance matrix is diagonal.
    # preserve only the diagonal of the covariance matrix
    diag_cov = np.zeros(covariances.shape)
    for i in range(10):
        for j in range(64):
            diag_cov[i, j, j] = covariances[i, j, j]
    print("with diagonal covariance matrix:")
    print("average conditional log-likelihood on train dataset is:",
          avg_conditional_likelihood(train_data, train_labels, means, diag_cov))
    print("average conditional log-likelihood on test dataset is:",
          avg_conditional_likelihood(test_data, test_labels, means, diag_cov))

    result_train_diag = classify_data(train_data, means, diag_cov)
    result_test_diag = classify_data(test_data, means, diag_cov)

    print("accuracy for train dataset is:", accuracy(result_train_diag, train_labels))
    print("accuracy for test dataset is:", accuracy(result_test_diag, test_labels))


if __name__ == "__main__":
    main()
