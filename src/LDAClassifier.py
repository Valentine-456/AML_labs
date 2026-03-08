from src.BaseClassifier import BaseClassifier
import numpy as np

class LDAClassifier(BaseClassifier):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.prior_proba0 = y[y == 0].size / y.size
        self.prior_proba1 = y[y == 1].size / y.size

        X0 = X[y == 0]
        X1 = X[y == 1]

        self.mu0 = X0.mean(axis=0)
        self.mu1 = X1.mean(axis=0)

        n0 = X0.shape[0]
        n1 = X1.shape[0]
        self.cov_matrix = (np.cov(X0.T) * n0 + np.cov(X1.T) * n1) / (n0 + n1)

        self.cov_matrix_inv = np.linalg.inv(self.cov_matrix)
        self.cov_matrix_det = np.linalg.det(self.cov_matrix)

    def _log_likelihood_for_features(self, X: np.ndarray, mu: np.ndarray):
        p = X.shape[1]
        deviation = (X - mu)
        mahalanobis_distance = np.sum((deviation @ self.cov_matrix_inv) * deviation, axis=1)
        return -0.5 * (mahalanobis_distance + p*np.log(2*np.pi) + np.log(self.cov_matrix_det)) 

    def predict_proba(self, Xtest: np.ndarray) -> np.ndarray:
        log_likelihood0 = self._log_likelihood_for_features(Xtest, self.mu0)
        score0 = log_likelihood0 + np.log(self.prior_proba0)

        log_likelihood1 = self._log_likelihood_for_features(Xtest, self.mu1)
        score1 = log_likelihood1 + np.log(self.prior_proba1)

        max_score = np.maximum(score0, score1)
        normalized_score0 = score0 - max_score
        normalized_score1 = score1 - max_score

        normalized_proba1 = np.exp(normalized_score1) / (np.exp(normalized_score1) + np.exp(normalized_score0))
        return normalized_proba1

    def predict(self, Xtest: np.ndarray) -> np.ndarray:
        return (self.predict_proba(Xtest) >= 0.5).astype(int)

    def get_params(self):
        return {
            "mu0": self.mu0, "mu1": self.mu1,
            "Prior probabilities": (self.prior_proba0, self.prior_proba1),
            "Covariance matrix": (self.cov_matrix),
            }
