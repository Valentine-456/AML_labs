import numpy as np


def train_test_split(X: np.ndarray, y: np.ndarray, ratio: float) -> tuple:
    n = X.shape[0]
    n_train = int(n * ratio)
    
    indices = np.random.permutation(n)
    
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
