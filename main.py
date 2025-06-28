import os

import feature_extraction
import processing
import data_io
import config

from model_tester import check_depth_sensitivity, evaluate_random_forest_max_features
from model_runner import run_decision_tree_pipeline, run_random_forest_pipeline, run_adaboost_pipeline
from sklearn_runner_DT import sklearn_model_DT

def pipeline():
    spam_data = feature_extraction.preprocessing()
    data_io.save_to_csv(spam_data)
    X_all = spam_data[config.SELECTED_FEATURES].astype(float).values
    feature_names = config.SELECTED_FEATURES
    X_train, X_test, y_train, y_test = processing.split(spam_data, X_all)
    X_train, y_train = processing.smote(X_train, y_train)
    if not os.path.isfile("models/decision_tree_model.pkl"):
        run_decision_tree_pipeline(X_train, X_test, y_train, y_test, ["ham", "spam"], feature_names)
        check_depth_sensitivity(X_train, X_test, y_train, y_test)
        sklearn_model_DT(X_train, X_test, y_train, y_test, target_names=["ham", "spam"], output_dir="models/sklearn_decision_tree_model.pkl")
    if not os.path.isfile("models/random_forest_model.pkl"):
        run_random_forest_pipeline(X_train, X_test, y_train, y_test, ["ham", "spam"])
        evaluate_random_forest_max_features(X_train, X_test, y_train, y_test)
    run_adaboost_pipeline(X_train, X_test, y_train, y_test, target_names=["ham", "spam"])

if __name__ == "__main__":
    pipeline()
