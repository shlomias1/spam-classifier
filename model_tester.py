import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier
from utils.plotting import plot_depth_sensitivity, plot_max_features_tuning
from utils.logger import _create_log
from processing import to_ndarray
import config

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

def evaluate_random_forest_max_features(X_train, X_test, y_train, y_test):
    X_train = to_ndarray(X_train)
    X_test = to_ndarray(X_test)
    max_features_range = range(2, 8)
    precisions = []
    recalls = []
    f1_scores = []
    for max_feats in max_features_range:
        _create_log(f"Training forest with max_features = {max_feats}", "info", "random_forest_max_features.log")
        clf = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS_RF,
            max_depth=config.MAX_DEPTH_RF,
            min_samples_split=config.MIN_SAMPLES_SPLIT_RF,
            max_features=max_feats,
            verbose=False
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
    plot_max_features_tuning(max_features_range, precisions, recalls, f1_scores, "random_forest_max_features_tuning.png")
