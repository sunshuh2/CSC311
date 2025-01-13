from check_grad import check_grad
from utils import load_train, load_train_small, load_valid, load_test
from logistic import logistic, logistic_predict, evaluate
import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    train_inputs_small, train_targets_small = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.4,
        "weight_regularization": 0.0,
        "num_iterations": 18,
    }
    weights = np.zeros((M + 1, 1))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    # YOUR CODE BEGINS HERE

    test_inputs, test_targets = load_test()

    ce_valid = []
    ce_train = []
    index = [i + 1 for i in range(hyperparameters["num_iterations"])]
    # on mnist_train
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        for i in range(M + 1):
            weights[i][0] = weights[i][0] - (hyperparameters["learning_rate"] * df[i][0])
        predict_valid = logistic_predict(weights, valid_inputs)
        ce, frac = evaluate(valid_targets, predict_valid)
        ce_valid.append(ce)
        predict_train = logistic_predict(weights, train_inputs)
        ce2, frac2 = evaluate(train_targets, predict_train)
        ce_train.append(ce2)

    plt.plot(index, ce_valid,  color='blue', label="valid")
    plt.plot(index, ce_train,  color='green', label="train")
    plt.title("mnist_train")
    plt.legend()
    plt.show()

    print("best-hyperparameters:", hyperparameters)  # several combinations give the same test error, this is chosen
    # based on the smallest number of iterations

    # for the best hyperparameters, report the ce and ca for train, valid and test
    input_sets = [train_inputs, valid_inputs, test_inputs]
    target_sets = [train_targets, valid_targets, test_targets]
    title = ["train", "valid", "test"]
    for i in range(3):
        predict = logistic_predict(weights, input_sets[i])
        ce, frac = evaluate(target_sets[i], predict)
        print("for", title[i], "cross entropy:", ce, "classification accuracy:", frac)
        if i == 1:
            print("test_error based on validation set:", 1 - frac)

    # for mnist_train_small
    weights = np.zeros((M + 1, 1))
    ce_valid_small = []
    ce_train_small = []
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs_small, train_targets_small, hyperparameters)
        for i in range(M + 1):
            weights[i][0] = weights[i][0] - (hyperparameters["learning_rate"] * df[i][0])
        predict_valid = logistic_predict(weights, valid_inputs)
        ce, frac = evaluate(valid_targets, predict_valid)
        ce_valid_small.append(ce)
        predict_train = logistic_predict(weights, train_inputs)
        ce2, frac2 = evaluate(train_targets, predict_train)
        ce_train_small.append(ce2)

    plt.plot(index, ce_valid_small,  color='blue', label="valid")
    plt.plot(index, ce_train_small, color='green', label="train")
    plt.title("mnist_train_small")
    plt.legend()
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic, weights, 0.001, data, targets, hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
