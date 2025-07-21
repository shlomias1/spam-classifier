from sklearn.tree import DecisionTreeClassifier as SklearnDT
from sklearn.metrics import accuracy_score, classification_report
from utils.logger import _create_log
from utils.plotting import save_confusion_matrix
import config
from data_io import save_model
from processing import to_ndarray

def sklearn_model_DT(X_train, X_test, y_train, y_test, target_names=None, output_dir="output"):
    _create_log("Running sklearn DecisionTreeClassifier...", "info", "sklearn_dt.log")

    # Train sklearn model
    clf = SklearnDT(max_depth=config.MAX_DEPTH_DT, min_samples_split=config.MIN_SAMPLES_SPLIT_DT, random_state=42)
    X_train = to_ndarray(X_train)
    X_test = to_ndarray(X_test)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names or ["class 0", "class 1"])
    _create_log(f"Sklearn Accuracy: {acc:.4f}\n{report}", "info", "sklearn_dt.log")

    # Save confusion matrix
    save_confusion_matrix(y_test, y_pred, labels=["ham", "spam"], path="confusion_matrix_sklearn_DT.png")
    
    # Save the model
    save_model(clf, output_dir)
    return acc, report