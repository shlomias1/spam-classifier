from sklearn.metrics import accuracy_score, classification_report
from decision_tree import DecisionTreeClassifier
from random_forest import RandomForestClassifier
from processing import to_ndarray
import config
from utils.plotting import save_confusion_matrix, plot_loss_trace, plot_feature_importance
from utils.logger import _create_log
from data_io import save_model

def run_decision_tree_pipeline(X_train, X_test, y_train, y_test, target_names=None, feature_names=None):
    _create_log("Starting decision tree training...","info","decision_tree_log.log")
    X_train = to_ndarray(X_train)
    X_test = to_ndarray(X_test)
    tree = DecisionTreeClassifier(
        max_depth=config.MAX_DEPTH_DT, 
        min_samples_split=config.MIN_SAMPLES_SPLIT_DT, 
        min_gain=1e-4,
        verbose=True)

    _create_log("Fitting model...","info","decision_tree_log.log")
    tree.fit(X_train, y_train)
    
    _create_log("Tree structure: ","info","decision_tree_log.log")
    tree.print_tree(feature_names)
    counter = tree.compute_feature_importance()
    plot_feature_importance(counter, feature_names, output_path="feature_importance_DT.png")
    for idx, count in counter.most_common(10):
        feature_label = feature_names[idx] if idx < len(feature_names) else f"X[{idx}]"
        _create_log(f"Feature '{feature_label}' used {count} times in splits", "info", "decision_tree_log.log")

    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    _create_log("Predicting on test set..","info","decision_tree_log.log")
    save_confusion_matrix(y_test, y_pred, labels=["ham", "spam"], path="confusion_matrix_DT.png")
    plot_loss_trace(tree.entropy_trace, path="loss_trace_DT.png", title="Decision Tree Loss Trace")

    _create_log(f"Accuracy: {acc:.4f}\nClassification report: ","info","decision_tree_log.log")
    _create_log(classification_report(y_test, y_pred, target_names=target_names or ["class 0", "class 1"]),"info","decision_tree_log.log")
    
    save_model(tree, "models/decision_tree_model.pkl")

def run_random_forest_pipeline(X_train, X_test, y_train, y_test, target_names=None, feature_names=None):
    _create_log("Starting Random Forest training...", "info", "random_forest_log.log")
    
    X_train = to_ndarray(X_train)
    X_test = to_ndarray(X_test)

    forest = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS_RF,
        max_depth=config.MAX_DEPTH_RF,
        min_samples_split=config.MIN_SAMPLES_SPLIT_RF,
        max_features=config.MAX_FEATURES_RF,
        verbose=True
    )

    _create_log("Fitting Random Forest model...", "info", "random_forest_log.log")
    forest.fit(X_train, y_train)

    _create_log("Predicting on test set...", "info", "random_forest_log.log")
    y_pred = forest.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    _create_log(f"Accuracy: {acc:.4f}", "info", "random_forest_log.log")
    _create_log("Classification report:\n" +
                classification_report(y_test, y_pred, target_names=target_names or ["class 0", "class 1"]),
                "info", "random_forest_log.log")

    save_confusion_matrix(y_test, y_pred, labels=target_names or ["class 0", "class 1"],
                          path="confusion_matrix_RF.png")

    save_model(forest, "models/random_forest_model.pkl")