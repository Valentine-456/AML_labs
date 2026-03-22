from abc import ABC, abstractmethod

class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict_proba(self, Xtest):
        pass

    @abstractmethod
    def predict(self, Xtest):
        pass

    @abstractmethod
    def get_params(self):
        pass