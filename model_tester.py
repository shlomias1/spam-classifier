import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
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

def compare_models(X_test, y_test, *model_entries):
    plt.figure(figsize=(10, 6))
    for name, model, score_method, label_format in model_entries:
        y_pred = model.predict(X_test)
        if hasattr(model, score_method):
            scorer = getattr(model, score_method)
            y_score = scorer(X_test)
            if score_method == 'predict_proba':
                y_score = y_score[:, 1]  
        else:
            raise ValueError(f"Model {name} missing method: {score_method}")
        if label_format == 'plusminus':
            y_true = np.where(y_test == 1, 1, -1)
        else:
            y_true = y_test

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("images/roc_comparison.png")
    plt.show()

