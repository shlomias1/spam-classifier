import numpy as np
from utils.logger import _create_log

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, min_gain=1e-7, verbose=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.verbose = verbose
        self.tree_ = None
        self.entropy_trace = []

    def fit(self, X, y):
        self.tree_ = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _best_split(self, X, y):
        best_gain = 0
        best_split = None
        current_entropy = self._entropy(y)
        n_samples, n_features = X.shape
        for j in range(n_features):
            thresholds = np.unique(X[:, j])
            for t in thresholds:
                _, y_left, _, y_right = self._split(X, y, j, t)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                p_left = len(y_left) / len(y)
                p_right = 1 - p_left
                gain = current_entropy - (p_left * self._entropy(y_left) + p_right * self._entropy(y_right))
                if gain > best_gain:
                    best_gain = gain
                    best_split = {'feature': j, 'threshold': t}
        return best_split, best_gain

    def _most_common_label(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _build_tree(self, X, y, depth=0):
        self.entropy_trace.append(self._entropy(y))
        if len(set(y)) == 1 or len(y) < self.min_samples_split or depth >= self.max_depth:
            return {'leaf': True, 'class': self._most_common_label(y)}
        split, best_gain = self._best_split(X, y)
        if split is None or best_gain < self.min_gain:
            return {'leaf': True, 'class': self._most_common_label(y)}
        if best_gain < self.min_gain:
            return {'leaf': True, 'class': self._most_common_label(y)}
        j, t = split['feature'], split['threshold']
        X_left, y_left, X_right, y_right = self._split(X, y, j, t)
        return {
            'leaf': False,
            'feature': j,
            'threshold': t,
            'left': self._build_tree(X_left, y_left, depth + 1),
            'right': self._build_tree(X_right, y_right, depth + 1)
        }

    def _predict_one(self, x, tree):
        if tree['leaf']:
            return tree['class']
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])

    def print_tree(self, tree=None, depth=0):
        if tree is None:
            tree = self.tree_

        indent = "  " * depth
        if tree['leaf']:
            _create_log(f"{indent}Leaf: class = {tree['class']}","info","decision_tree_log.log")
        else:
            feature = tree['feature']
            threshold = tree['threshold']
            _create_log(f"{indent}Node: feature = X[{feature}], threshold = {threshold}","info","decision_tree_log.log")
            self.print_tree(tree['left'], depth + 1)
            self.print_tree(tree['right'], depth + 1)