import os

import feature_extraction
import processing
import data_io
import config

from model_tester import check_depth_sensitivity, evaluate_random_forest_max_features
from model_runner import run_decision_tree_pipeline, run_random_forest_pipeline, run_adaboost_pipeline
from sklearn_runner_DT import sklearn_model_DT
from utils.plotting import plot_roc_curves_comparison

def pipeline():
    if not os.path.isfile('data/spam_processing_data.csv'):
        spam_data = feature_extraction.preprocessing()
        data_io.save_to_csv(spam_data)
    spam_data = data_io.load_data('data/spam_processing_data.csv')
    
    X_all = spam_data[config.SELECTED_FEATURES].astype(float).values
    feature_names = config.SELECTED_FEATURES
    X_train, X_test, y_train, y_test = processing.split(spam_data, X_all)
    X_train, y_train = processing.smote(X_train, y_train)
    
    if not os.path.isfile("models/decision_tree_model.pkl"):
        check_depth_sensitivity(X_train, X_test, y_train, y_test)
        run_decision_tree_pipeline(X_train, X_test, y_train, y_test, ["ham", "spam"], feature_names)
        sklearn_model_DT(X_train, X_test, y_train, y_test, target_names=["ham", "spam"], output_dir="models/sklearn_decision_tree_model.pkl")
    if not os.path.isfile("models/random_forest_model.pkl"):
        evaluate_random_forest_max_features(X_train, X_test, y_train, y_test)
        run_random_forest_pipeline(X_train, X_test, y_train, y_test, ["ham", "spam"])
    if not os.path.isfile("models/adaboost_model.pkl"):
        run_adaboost_pipeline(X_train, X_test, y_train, y_test, target_names=["ham", "spam"])
    
    rf_model = data_io.load_model("models/random_forest_model.pkl")
    dt_model = data_io.load_model("models/decision_tree_model.pkl")
    ada_model = data_io.load_model("models/adaboost_model.pkl")
    
    rf_score = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = rf_model.predict(X_test)
    dt_score = dt_model.predict_proba(X_test)[:, 1]
    dt_pred = dt_model.predict(X_test)
    ada_score = ada_model.decision_function(X_test) 
    ada_pred = ada_model.predict(X_test)
    
    plot_roc_curves_comparison(
        models_outputs=[
            ("Decision Tree", dt_pred, dt_score, "binary"),
            ("Random Forest", rf_pred, rf_score, "binary"),
            ("AdaBoost", ada_pred, ada_score, "plusminus"),
        ],
        y_test=y_test
    )

if __name__ == "__main__":
    pipeline()
