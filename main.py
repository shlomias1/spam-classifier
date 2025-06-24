import feature_extraction
import processing
import data_io
from sklearn.metrics import accuracy_score, classification_report

from model_runner import run_decision_tree_pipeline
import config

def pipeline():
    spam_data = feature_extraction.preprocessing()
    data_io.save_to_csv(spam_data)
    X_all, tfidf_vectorizer = processing.tf_idf_vectorization(spam_data)
    X_train, X_test, y_train, y_test = processing.split(spam_data, X_all)
    data_io.save_processed_data(X_train, X_test, y_train, y_test, tfidf_vectorizer, filename_prefix='data/processed_data')
    run_decision_tree_pipeline(X_train, X_test, y_train, y_test, target_names=["ham", "spam"])

if __name__ == "__main__":
    pipeline()
