from sklearn.metrics import accuracy_score, classification_report
from decision_tree import DecisionTreeClassifier
from processing import to_ndarray
import config
from utils.plotting import save_confusion_matrix, plot_loss_trace
from utils.logger import _create_log
from data_io import save_model

def run_decision_tree_pipeline(X_train, X_test, y_train, y_test, target_names=None):
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
    tree.print_tree()

    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    _create_log("Predicting on test set..","info","decision_tree_log.log")
    save_confusion_matrix(y_test, y_pred, labels=["ham", "spam"], path="confusion_matrix_DT.png")
    plot_loss_trace(tree.entropy_trace, path="loss_trace_DT.png", title="Decision Tree Loss Trace")

    _create_log(f"Accuracy: {acc:.4f}\nClassification report: ","info","decision_tree_log.log")
    _create_log(classification_report(y_test, y_pred, target_names=target_names or ["class 0", "class 1"]),"info","decision_tree_log.log")
    
    save_model(tree, "models/decision_tree_model.pkl")