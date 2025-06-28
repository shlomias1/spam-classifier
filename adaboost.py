import numpy as np
from decision_stump import DecisionStump

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float('inf')

            # Scan on each feature and threshold
            for feature_i in range(n_features):
                feature_values = X[:, feature_i]
                thresholds = np.unique(feature_values)
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[feature_values < threshold] = -1
                        else:
                            predictions[feature_values > threshold] = -1
                        # Error calculation
                        miss = predictions != y
                        error = np.sum(w[miss])
                        if error < min_error:
                            stump.polarity = polarity
                            stump.threshold = threshold
                            stump.feature_index = feature_i
                            min_error = error
            # Expertise calculation
            EPS = 1e-10
            stump.alpha = 0.5 * np.log((1 - min_error + EPS) / (min_error + EPS))
            # Update weights
            predictions = stump.predict(X)
            w *= np.exp(-stump.alpha * y * predictions)
            w /= np.sum(w)  # Normalization
            self.models.append(stump)
            self.alphas.append(stump.alpha)

    def predict(self, X):
        clf_preds = [model.alpha * model.predict(X) for model in self.models]
        return np.sign(np.sum(clf_preds, axis=0))
