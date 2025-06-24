from sklearn.metrics import accuracy_score, classification_report
from decision_tree import DecisionTreeClassifier
from processing import to_ndarray
import config

def run_decision_tree_pipeline(X_train, X_test, y_train, y_test, target_names=None):
    print("Starting decision tree training...")
    X_train = to_ndarray(X_train)
    X_test = to_ndarray(X_test)

    tree = DecisionTreeClassifier(
        max_depth=config.MAX_DEPTH_DT, 
        min_samples_split=config.MIN_SAMPLES_SPLIT_DT, 
        min_gain=1e-4,
        verbose=True)

    print("Fitting model...")
    tree.fit(X_train, y_train)
    
    print("Tree structure:")
    tree.print_tree()

    print("\nPredicting on test set...")
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names or ["class 0", "class 1"]))