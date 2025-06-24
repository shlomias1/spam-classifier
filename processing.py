from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import numpy as np

import config

# TF-IDF Vectorization for textual input
def tf_idf_vectorization(data):
    # Initialize the TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=config.MAX_FEATURE_TF_IDF, ngram_range=(1, 2), stop_words='english')
    # Fit and transform the cleaned text data
    X_tfidf = tfidf.fit_transform(data['cleaned'])
    # Features to use for tree-based model
    X_manual = data[config.SELECTED_FEATURES].astype(float).values
    # Final design matrix
    X_all = hstack([X_tfidf, X_manual])
    return X_all, tfidf

def split(data, X_all):
    # splitting the dataset into training and testing sets
    y = data['label'].map({'ham': 0, 'spam': 1})
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, stratify=y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def to_ndarray(X):
    if not isinstance(X, np.ndarray):
        return X.toarray()
    else:
        return X 
