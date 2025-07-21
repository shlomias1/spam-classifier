import numpy as np
from collections import Counter
from utils.logger import _create_log
from decision_tree import DecisionTreeClassifier
from config import SELECTED_FEATURES

class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, max_features=None, verbose=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.verbose = verbose 
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True) 
            X_sample = X[indices]
            y_sample = np.array(y)[indices]
            if self.max_features is None: 
                max_feats = n_features
            elif self.max_features == "sqrt":
                max_feats = int(np.sqrt(n_features))
            elif isinstance(self.max_features, float):
                max_feats = int(self.max_features * n_features)
            else:
                max_feats = int(self.max_features)
            feature_indices = np.random.choice(n_features, max_feats, replace=False)
            selected_names = [SELECTED_FEATURES[i] for i in feature_indices]
            X_sample_reduced = X_sample[:, feature_indices]
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                verbose=self.verbose
            )
            tree.fit(X_sample_reduced, y_sample)
            self.trees.append((tree, feature_indices))
            _create_log(
                f"Trained tree {i+1}/{self.n_estimators} with {len(feature_indices)} features: {selected_names}",
                "info",
                "random_forest_log.log"
            )

    def predict(self, X):
        all_preds = []
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_input = X if feature_indices is None else X[:, feature_indices]
            preds = tree.predict(X_input)
            all_preds.append(preds)
            _create_log(f"Predictions from tree {i+1}: {np.bincount(preds.astype(int))}", "info", "random_forest_log.log")
        all_preds = np.array(all_preds)
        final_preds = np.round(np.mean(all_preds, axis=0)).astype(int)
        return final_preds

    def predict_proba(self, X):
        all_preds = []
        for tree, feature_indices in self.trees:
            X_input = X if feature_indices is None else X[:, feature_indices]
            preds = tree.predict(X_input)
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        proba_class_1 = np.mean(all_preds, axis=0)  
        proba_class_0 = 1 - proba_class_1            
        return np.vstack((proba_class_0, proba_class_1)).T

    def compute_feature_importance(self):
        importance_counter = Counter()
        for tree, feature_indices in self.trees:
            local_importance = tree.compute_feature_importance()
            if feature_indices is not None:
                mapped = {feature_indices[i]: count for i, count in local_importance.items()}
                importance_counter.update(mapped)
            else:
                importance_counter.update(local_importance)
        return importance_counter

    def log_feature_importance(self, feature_names=None, top_n=10):
        importance = self.compute_feature_importance()
        for idx, count in importance.most_common(top_n):
            name = feature_names[idx] if feature_names else f"X[{idx}]"
            _create_log(f"Feature {name} used {count} times in splits across trees", "info", "random_forest_log.log")
