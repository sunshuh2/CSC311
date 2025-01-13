from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree, export_graphviz
import matplotlib.pyplot as plt
import math


def load_data():
    headlines = []
    f_real = open("clean_real.txt")
    headline = f_real.readline()
    while headline != "":
        headlines.append(headline[:-1])
        headline = f_real.readline()
    f_real.close()
    f_fake = open("clean_fake.txt")
    headline = f_fake.readline()
    while headline != "":
        headlines.append(headline[:-1])
        headline = f_fake.readline()
    f_fake.close()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(headlines)
    features = vectorizer.get_feature_names_out()
    y = [1] * 1968 + [0] * 1298  # 1 for real, 0 for fake
    X_train, X_1, y_train, y_1 = train_test_split(X, y, train_size=0.7)
    X_val, X_test, y_val, y_test = train_test_split(X_1, y_1, train_size=0.5)
    return X_train, X_val, X_test, y_train, y_val, y_test, features, X, y


def accuracy(y, y_predict):
    N = len(y)
    summation = 0
    for i in range(len(y)):
        if y[i] == y_predict[i]:
            summation += 1
    return summation / N


def select_model(X_train, X_val, y_train, y_val, feature_names):
    max_depth = [5, 10, 15, 30, 50, 70, 100]
    ig_performance = []
    ll_performance = []
    gc_performance = []
    ig = []
    ll = []
    gc = []
    for d in max_depth:
        ig_classifier = DecisionTreeClassifier(criterion="entropy", max_depth=d)  # information gain
        ll_classifier = DecisionTreeClassifier(criterion="log_loss", max_depth=d)  # log loss
        gc_classifier = DecisionTreeClassifier(criterion="gini", max_depth=d)  # Gini coefficient
        ig_classifier.fit(X_train, y_train)
        ll_classifier.fit(X_train, y_train)
        gc_classifier.fit(X_train, y_train)
        ig_performance.append(accuracy(y_val, ig_classifier.predict(X_val)))
        ig.append(ig_classifier)
        ll_performance.append(accuracy(y_val, ll_classifier.predict(X_val)))
        ll.append(ll_classifier)
        gc_performance.append(accuracy(y_val, gc_classifier.predict(X_val)))
        gc.append(gc_classifier)
    print(ig_performance, ll_performance,  gc_performance)
    plt.plot(max_depth, ig_performance, color='red', label="information gain")
    plt.plot(max_depth, ll_performance, color='blue', label="log loss")
    plt.plot(max_depth, gc_performance, color='green', label="Gini coefficient")
    plt.legend()
    plt.show()
    max_accuracy = max(ig_performance + ll_performance + gc_performance)
    classifier = (ig + ll + gc)[(ig_performance + ll_performance + gc_performance).index(max_accuracy)]
    export_graphviz(classifier, out_file="tree.dot", feature_names=feature_names, filled=True, rounded=True,
                    special_characters=True, max_depth=2)
    # method of using export_graphviz and Graphviz to make this tree.dot file into png was from chatgpt


def entropy(p_Y):  # p_Y = [p(y), y \in Y]
    s = 0
    for p in p_Y:
        if p != 0:
            s += p * math.log(p, 2)
    return -s


def compute_information_gain(X, y, feature_names, feature):  # IG(Y|X=x_i) = H(Y) - H(Y|X=x_i)
    all_count = len(y)
    y_count = [0, 0]
    for y_i in y:
        y_count[y_i] += 1
    H_Y = entropy([y_count[1] / all_count, y_count[0] / all_count])  # H(Y)
    index = feature_names.index(feature)
    real_fake = {1: {}, 0: {}}
    for i in range(len(y)):
        x_i = X[i, index]  # from chatgpt, originally X[i][index], changed to deal with sparse matrix
        if x_i not in real_fake[y[i]]:
            real_fake[y[i]][x_i] = 1
        else:
            real_fake[y[i]][x_i] += 1
    # real_fake = {real:{occurrence of feature: frequency of occurrence}}
    possible_x = list(set().union(list(real_fake[1].keys()), list(real_fake[0].keys())))  # possible frequency of the
    # feature
    for x in possible_x:
        if x not in real_fake[1]:
            real_fake[1][x] = 0
        if x not in real_fake[0]:
            real_fake[0][x] = 0
    H_YX = 0  # H(Y|X) = \sum_{x \in X} p(x) H(Y|X=x)
    for x in possible_x:
        x_count = real_fake[1][x] + real_fake[0][x]
        H_yx = entropy([real_fake[1][x] / x_count, real_fake[0][x] / x_count])  # compute H(Y|X=x)
        p_x = (real_fake[1][x] + real_fake[0][x]) / all_count  # p(x)
        H_YX += (p_x * H_yx)
    return H_Y - H_YX


X_train, X_val, X_test, y_train, y_val, y_test, features, X, y = load_data()
select_model(X_train, X_val, y_train, y_val, features)

top_features = ["the", "hillary", "trump", "donald", "year", "market"]
for f in top_features:
    print(compute_information_gain(X_train, y_train, list(features), f))
