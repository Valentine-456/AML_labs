from src.BaseClassifier import BaseClassifier
import numpy as np

class NBClassifier(BaseClassifier):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.prior_proba0 = y[y == 0].size / y.size
        self.prior_proba1 = y[y == 1].size / y.size

        X0 = X[y == 0]
        X1 = X[y == 1]
        self.mu0 = X0.mean(axis=0)
        self.mu1 = X1.mean(axis=0)
        self.var0 = X0.var(axis=0)
        self.var1 = X1.var(axis=0)

    def _log_likelihood_for_features(self, X: np.ndarray, mu: np.ndarray, var: np.ndarray):
        log_gauss_pdf_value = -0.5 * np.log(2 * np.pi * var) - ((X - mu)**2) / (2 * var)
        return np.sum(log_gauss_pdf_value, axis=1)


    def predict_proba(self, Xtest: np.ndarray) -> np.ndarray:
        log_likelihood0 = self._log_likelihood_for_features(Xtest, self.mu0, self.var0)
        score0 = log_likelihood0 + np.log(self.prior_proba0)

        log_likelihood1 = self._log_likelihood_for_features(Xtest, self.mu1, self.var1)
        score1 = log_likelihood1 + np.log(self.prior_proba1)

        max_score = np.maximum(score0, score1)
        normalized_score0 = score0 - max_score
        normalized_score1 = score1 - max_score

        normalized_proba1 = np.exp(normalized_score1) / (np.exp(normalized_score1) + np.exp(normalized_score0))
        return normalized_proba1

    def predict(self, Xtest: np.ndarray) -> np.ndarray:
        return (self.predict_proba(Xtest) >= 0.5).astype(int)

    def get_params(self):
        return {"mu0": self.mu0, "mu1": self.mu1,
                "Prior probabilities": (self.prior_proba0, self.prior_proba1)}

