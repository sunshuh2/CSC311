from datetime import datetime

from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import os
import csv
import ast
from utils import (
    load_valid_csv,
    load_public_test_csv,
    load_train_csv
)
from typing import Optional

np.random.seed(821)
Array = np.array
simple_keywords = ['Basic', 'Addition', 'Adding', 'Subtraction', 'Subtracting', 'Ordering', 'Written']
hard_keywords = ['Advanced', 'Proof', 'Theorem', 'Theorems', 'Functions']


def load_subject_difficulty(root_dir="./data"):
    """ return a dictionary with subject id as key and difficulty from [1,2,3]"""
    path = os.path.join(root_dir, "subject_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                if any(keyword in row[1] for keyword in hard_keywords):
                    data[int(row[0])] = 3
                elif any(keyword in row[1] for keyword in simple_keywords):
                    data[int(row[0])] = 1
                else:
                    data[int(row[0])] = 2
            except ValueError:
                # Pass first row.
                pass
    return data


def load_question_difficulty(root_dir="./data"):
    """ return a dictionary with question id as key and difficultly from [1,2,3]"""
    subject = load_subject_difficulty(root_dir)
    path = os.path.join(root_dir, "question_meta.csv")
    question_subject = {}
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
        # Initialize the data.
    data = {}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                question_subject[int(row[0])] = ast.literal_eval(row[1])
                subject_list = ast.literal_eval(row[1])
                max_difficulty = -1
                for sid in subject_list:
                    if max_difficulty < subject[sid]:
                        max_difficulty = subject[sid]
                data[int(row[0])] = max_difficulty
            except ValueError:
                # Pass first row.
                pass
    return data, question_subject


def load_student_age(root_dir="./data"):
    """ return a dictionary with student id as key and their age"""
    path = os.path.join(root_dir, "student_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
        # Initialize the data.
    data = {i: -1 for i in range(542)}
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                date_of_birth = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S.%f")
                current_date = datetime.now()
                age = current_date.year - date_of_birth.year
                if 0 < age <= 16:
                    data[int(row[0])] = 0
                elif age > 16:
                    data[int(row[0])] = 1
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # birthdate might not be available.
                pass
    return data


def bootstrap(X_train: Array, y_train: Array, samples_index: list[list[int]]):
    """ take the dataset, generate m new datasets with n examples, n same as the original size of train data. We """
    samples = []
    m = len(samples_index)
    for i in range(m):
        sample_index = samples_index[i]
        sample = [X_train[sample_index, :], y_train[sample_index]]
        samples.append(sample)
    return samples


def decision_trees(samples: list[Array, Array], max_depth: int):
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


def add_feature(id_list: list[int], meta: dict[int, Optional[int]]):
    """ return a list with the question feature corresponding to the question id """
    data = []
    for i in id_list:
        if i in meta:
            data.append(meta[i])
    return data


def major_vote(n: int, total: int):
    """ help function for major vote """
    if n >= total / 2:
        return 1
    else:
        return 0


def past_history(id_list: list[int], y: list[int], n: int):
    """ if the id is student id, then return two dictionaries dict[student_id: the number of correct answer/completed
    question for each student], if the id is question id, then return two dictionaries dict[question_id: the number of
    student correctly answered it/ completed it """
    completion = {i: 0 for i in range(n)}
    correct = {i: 0 for i in range(n)}
    for i in range(len(id_list)):
        completion[id_list[i]] += 1
        correct[id_list[i]] += y[i]
    return correct, completion  # these two attributes gives the most improvements


def evaluate_subject(student: list[int], question: list[int], q_to_s: dict[int, list[int]], y: list[int]):
    """ return count of completion and correctness for each student and each subject"""
    subject_correct = {i: [0 for i in range(388)] for i in range(542)}
    for i in range(len(student)):
        subject_list = q_to_s[question[i]]
        for s in subject_list:
            subject_correct[student[i]][s] += y[i]
    return subject_correct


def main():
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    m = 5

    difficulty, question_subject = load_question_difficulty("./data")
    age = load_student_age("./data")
    student_correct, student_completion = past_history(train_data["user_id"], train_data["is_correct"], 542)
    question_correct, question_completion = past_history(train_data["question_id"], train_data["is_correct"], 1774)

    difficulty_train = add_feature(train_data["question_id"], difficulty)
    age_train = add_feature(train_data["user_id"], age)
    s_past_correct = add_feature(train_data["user_id"], student_correct)
    s_past_completion = add_feature(train_data["user_id"], student_completion)
    q_past_correct = add_feature(train_data["question_id"], question_correct)
    q_past_completion = add_feature(train_data["question_id"], question_completion)

    subject_correctness = evaluate_subject(train_data["user_id"], train_data["question_id"],
                                           question_subject, train_data["is_correct"])

    X_train = np.array([train_data["question_id"], train_data["user_id"], difficulty_train, age_train,
                        s_past_correct, s_past_completion, q_past_correct, q_past_completion]).T
    y_train = np.array(train_data["is_correct"])

    scaler = StandardScaler()
    subject_info = np.array(add_feature(train_data["user_id"], subject_correctness))
    X_train_pca = np.concatenate((X_train, subject_info), axis=1)
    scaler.fit(X_train_pca)
    X_train_scaled = scaler.transform(X_train_pca)
    pca = PCA(0.98)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # We want to check whether these (new) features give reasonable amount of information
    feature_names = ["question_id", "user_id", "difficulty", "age", "student past correct", "student past completion",
                     "question past correct", "question past completion"]
    res = dict(zip(feature_names, mutual_info_classif(X_train, y_train, discrete_features=True)))
    print(res)

    N = len(y_train)
    samples = [np.random.randint(low=0, high=N, size=N) for i in range(m)]
    train_samples = bootstrap(X_train, y_train, samples)
    models = decision_trees(train_samples, 14)
    train_samples_pca = bootstrap(X_train_pca, y_train, samples)
    models_pca = decision_trees(train_samples_pca, 15)

    # add new features for valid data
    difficulty_val = add_feature(val_data["question_id"], difficulty)
    age_val = add_feature(val_data["user_id"], age)
    s_past_correct = add_feature(val_data["user_id"], student_correct)
    s_past_completion = add_feature(val_data["user_id"], student_completion)
    q_past_correct = add_feature(val_data["question_id"], question_correct)
    q_past_completion = add_feature(val_data["question_id"], question_completion)
    X_val = np.array([val_data["question_id"], val_data["user_id"], difficulty_val, age_val,
                      s_past_correct, s_past_completion, q_past_correct, q_past_completion]).T
    y_val = np.array(val_data["is_correct"])

    subject_info = np.array(add_feature(val_data["user_id"], subject_correctness))
    X_val_pca = np.concatenate((X_val, subject_info), axis=1)
    X_val_scaled = scaler.transform(X_val_pca)
    X_val_pca = pca.transform(X_val_scaled)

    # add features for test data
    difficulty_test = add_feature(test_data["question_id"], difficulty)
    age_test = add_feature(test_data["user_id"], age)
    s_past_correct = add_feature(test_data["user_id"], student_correct)
    s_past_completion = add_feature(test_data["user_id"], student_completion)
    q_past_correct = add_feature(test_data["question_id"], question_correct)
    q_past_completion = add_feature(test_data["question_id"], question_completion)
    X_test = np.array([test_data["question_id"], test_data["user_id"], difficulty_test, age_test,
                       s_past_correct, s_past_completion, q_past_correct, q_past_completion]).T
    y_test = np.array(test_data["is_correct"])

    subject_info = np.array(add_feature(test_data["user_id"], subject_correctness))
    X_test_pca = np.concatenate((X_test, subject_info), axis=1)
    X_test_scaled = scaler.transform(X_test_pca)
    X_test_pca = pca.transform(X_test_scaled)

    predictions = []
    ensemble_val = [0 for i in range(len(X_val))]
    ensemble_test = [0 for i in range(len(X_test))]
    ensemble_train = [0 for i in range(len(X_train))]
    for i in range(m):
        model = models[i]
        export_graphviz(model, out_file=f"tree{i}.dot", feature_names=feature_names, filled=True, rounded=True,
                        special_characters=True, max_depth=2)
        prediction = model.predict(X_val)
        predictions.append(prediction)
        # print(f"Accuracy for base model {i + 1} on validation set:", accuracy(prediction, y_val))
        ensemble_val += prediction
        ensemble_test += model.predict(X_test)
        ensemble_train += model.predict(X_train)
    ensemble_val = [major_vote(i, m) for i in ensemble_val]
    ensemble_test = [major_vote(i, m) for i in ensemble_test]
    ensemble_train = [major_vote(i, m) for i in ensemble_train]
    print("Accuracy for ensemble model on train set:", accuracy(np.array(ensemble_train), y_train))
    print("Accuracy for ensemble model on validation set:", accuracy(np.array(ensemble_val), y_val))
    print("Accuracy for ensemble model on test set:", accuracy(np.array(ensemble_test), y_test))

    predictions_pca = []
    ensemble_val = [0 for i in range(len(X_val))]
    ensemble_test = [0 for i in range(len(X_test))]
    ensemble_train = [0 for i in range(len(X_train))]
    for i in range(m):
        model = models_pca[i]
        prediction = model.predict(X_val_pca)
        predictions_pca.append(prediction)
        # print(f"Accuracy for base model {i + 1} on validation set with PCA:", accuracy(prediction, y_val))
        ensemble_val += prediction
        ensemble_test += model.predict(X_test_pca)
        ensemble_train += model.predict(X_train_pca)
    ensemble_val = [major_vote(i, m) for i in ensemble_val]
    ensemble_test = [major_vote(i, m) for i in ensemble_test]
    ensemble_train = [major_vote(i, m) for i in ensemble_train]
    print("Accuracy for ensemble model on train set with PCA:", accuracy(np.array(ensemble_train), y_train))
    print("Accuracy for ensemble model on validation set with PCA:", accuracy(np.array(ensemble_val), y_val))
    print("Accuracy for ensemble model on test set with PCA:", accuracy(np.array(ensemble_test), y_test))


if __name__ == "__main__":
    main()
