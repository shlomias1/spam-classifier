import pandas as pd
import numpy as np
import pickle
from utils.logger import _create_log
import os

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='latin-1')
    return data

def save_to_csv(data, filename='data/spam_processing_data.csv'):
    data.to_csv(filename, index=False)
    
def save_processed_data(X_train, X_test, y_train, y_test, tfidf_vectorizer, filename_prefix='processed_data'):
    np.savez_compressed(f'{filename_prefix}_train.npz', X=X_train, y=y_train)
    np.savez_compressed(f'{filename_prefix}_test.npz', X=X_test, y=y_test)
    with open(f'{filename_prefix}_tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)

def save_model(model, path):
    if not os.path.exists("models"):
        os.makedirs("models")
    with open(path, "wb") as f:
        pickle.dump(model, f)
