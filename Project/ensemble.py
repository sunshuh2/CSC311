from sklearn.tree import DecisionTreeClassifier

import numpy as np

from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_csv
)

Array = np.array
# np.random.seed(821)


def bootstrap(X_train: Array, y_train: Array, m: int):
    """ take the dataset, generate m new datasets with n examples, n same as the original size of train data"""
    N = len(y_train)
    samples = []
    for i in range(m):
        sample_index = np.random.randint(low=0, high=N, size=N)
        sample = [X_train[sample_index, :], y_train[sample_index]]
        samples.append(sample)
    return samples


def decision_trees(samples: list[Array], max_depth: int):
    """ create and fit m = len(samples) decision trees based on each bootstrap sample with max_depth given"""
    models = []
    for i in range(len(samples)):
        sample = samples[i]
        model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy')
        model.fit(sample[0], sample[1])
        models.append(model)
    return models


def accuracy(predictions: Array, y: Array):
    """ evaluate the accuracy of prediction on y """
    return np.mean(predictions == y)


def major_vote(n: int, total: int):
    """ help function for major vote """
    if n >= total/2:
        return 1
    else:
        return 0


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    m = 3

    X_train = np.array([train_data["question_id"], train_data["user_id"]]).T
    y_train = np.array(train_data["is_correct"])
    train_samples = bootstrap(X_train, y_train, m)
    models = decision_trees(train_samples, 16)

    X_val = np.array([val_data["question_id"], val_data["user_id"]]).T
    y_val = np.array(val_data["is_correct"])
    X_test = np.array([test_data["question_id"], test_data["user_id"]]).T
    y_test = np.array(test_data["is_correct"])

    predictions = []
    ensemble_val = [0 for i in range(len(X_val))]
    ensemble_test = [0 for i in range(len(X_test))]
    for i in range(m):
        model = models[i]
        prediction = model.predict(X_val)
        predictions.append(prediction)
        print(f"Accuracy for base model {i + 1} on validation set:", accuracy(prediction, y_val))
        ensemble_val += prediction
        ensemble_test += model.predict(X_test)
    ensemble_val = [major_vote(i, m) for i in ensemble_val]
    ensemble_test = [major_vote(i, m) for i in ensemble_test]
    print("Accuracy for ensemble model on validation set:", accuracy(np.array(ensemble_val), y_val))
    print("Accuracy for ensemble model on test set:", accuracy(np.array(ensemble_test), y_test))


if __name__ == "__main__":
    main()
