import pandas as pd
import numpy as np
import pickle

spam_data = pd.read_csv('data/spam.csv', encoding='latin-1')

def save_processed_data(X_train, X_test, y_train, y_test, tfidf_vectorizer, filename_prefix='processed_data'):
    np.savez_compressed(f'{filename_prefix}_train.npz', X=X_train, y=y_train)
    np.savez_compressed(f'{filename_prefix}_test.npz', X=X_test, y=y_test)
    with open(f'{filename_prefix}_tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)