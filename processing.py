from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix, issparse
import numpy as np
from imblearn.over_sampling import SMOTE

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
    tfidf_feature_names = tfidf.get_feature_names_out()
    manual_features = config.SELECTED_FEATURES
    all_feature_names = list(tfidf_feature_names) + manual_features
    return X_all, tfidf, all_feature_names

def split(data, X_all):
    data['label'] = data['label'].str.strip().str.lower()
    y = data['label'].map({'ham': 0, 'spam': 1})
    valid_indices = y.notna()
    y = y[valid_indices].reset_index(drop=True)
    if issparse(X_all):
        X_all = X_all[valid_indices.values]
    else:
        X_all = X_all[valid_indices.values, :]
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, stratify=y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def smote(X_train, y_train): 
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train

def to_ndarray(X):
    if not isinstance(X, np.ndarray):
        return X.toarray()
    else:
        return X