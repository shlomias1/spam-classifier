import feature_extraction
import processing
import data_io
from model_tester import check_depth_sensitivity
from model_runner import run_decision_tree_pipeline
from sklearn_runner_DT import sklearn_model_DT
import os

def pipeline():
    spam_data = feature_extraction.preprocessing()
    data_io.save_to_csv(spam_data)
    X_all, tfidf_vectorizer, feature_names = processing.tf_idf_vectorization(spam_data)
    X_train, X_test, y_train, y_test = processing.split(spam_data, X_all)
    data_io.save_processed_data(X_train, X_test, y_train, y_test, tfidf_vectorizer, filename_prefix='data/processed_data')
    # if not os.path.isfile("models/decision_tree_model.pkl"):
    #     run_decision_tree_pipeline(X_train, X_test, y_train, y_test, ["ham", "spam"], feature_names)
    #     check_depth_sensitivity(X_train, X_test, y_train, y_test)
    #     sklearn_model_DT(X_train, X_test, y_train, y_test, target_names=["ham", "spam"], output_dir="models/sklearn_decision_tree_model.pkl")
    run_decision_tree_pipeline(X_train, X_test, y_train, y_test, ["ham", "spam"], feature_names=feature_names)
    
if __name__ == "__main__":
    pipeline()