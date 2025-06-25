import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from decision_tree import DecisionTreeClassifier
from utils.plotting import plot_depth_sensitivity
from utils.logger import _create_log
from processing import to_ndarray

def check_depth_sensitivity(X_train, X_test, y_train, y_test):
    depth_range = range(2, 11)
    results = []
    for depth in depth_range:
        _create_log(f"Training Decision Tree with max_depth={depth}...", "info")
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_split=10)
        X_train = to_ndarray(X_train)
        X_test = to_ndarray(X_test)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        results.append((depth, precision, recall, f1))
    results_array = np.array(results)
    plot_depth_sensitivity(results_array, "depth_sensitivity.png")


