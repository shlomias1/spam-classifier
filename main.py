import feature_extraction
import processing
import data_io

def pipeline():
    spam_data = feature_extraction.preprocessing()
    X_all, tfidf_vectorizer = processing.tf_idf_vectorization(spam_data)
    X_train, X_test, y_train, y_test = processing.split(spam_data, X_all)
    data_io.save_processed_data(X_train, X_test, y_train, y_test, tfidf_vectorizer, filename_prefix='data/processed_data')

if __name__ == "__main__":
    pipeline()