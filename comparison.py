import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

ROOT = path.dirname(path.realpath(__file__))
RESOURCE = path.join(ROOT, 'resource')
MODEL_PATH = path.join(ROOT, 'models')
data_file = path.join(RESOURCE, 'tropic_huwz.xlsx')

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(5),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=3, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=7),
    RandomForestClassifier(max_depth=7, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


def generate_data_sets(excel_name, label_column, exclude_columns):
    raw_data = pd.read_excel(excel_name).values
    row_number, column_number = raw_data.shape
    x_columns = list(set(range(column_number)) - {label_column} - set(exclude_columns))
    return raw_data[:, x_columns], raw_data[:, label_column]


X, y = generate_data_sets(data_file, 13, [0, 2, 4, 5, 11, 12])
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)


for name, clf in zip(names, classifiers):
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print "score for %s is %f" % (name, score)
