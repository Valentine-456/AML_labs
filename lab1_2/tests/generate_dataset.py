import numpy as np

def generate_dataset(n: int, a: float):
    X = np.array([
        np.random.normal(loc=0, scale=1, size=n),
        np.random.normal(loc=a, scale=1, size=n),
    ], ).T
    y = np.array(np.random.binomial(n=1, p=0.5, size=n)).T
    return (X, y)


def generate_dataset_scheme1(a: float, n=1000):
    y = np.array(np.random.binomial(n=1, p=0.5, size=n))

    X = np.zeros(shape=(y.shape[0], 2))
    cov_matrix = np.eye(2)

    X[y == 0] = np.random.multivariate_normal([0, 0], cov_matrix, size=(y==0).sum())
    X[y == 1] = np.random.multivariate_normal([a, a], cov_matrix, size=(y==1).sum())
    return X, y

def generate_dataset_scheme2(a: float, correlation_coef, n=1000):
    y = np.array(np.random.binomial(n=1, p=0.5, size=n))

    X = np.zeros(shape=(y.shape[0], 2))

    mean0 = np.array([0, 0])
    cov_matrix0 = np.array([
        [1, correlation_coef],
        [correlation_coef, 1]
    ])
    X[y == 0] = np.random.multivariate_normal(mean=mean0, cov=cov_matrix0, size=(y==0).sum())

    mean1 = np.array([a, a])
    cov_matrix1 = np.array([
        [1, -correlation_coef],
        [-correlation_coef, 1]
    ])
    X[y == 1] = np.random.multivariate_normal(mean=mean1, cov=cov_matrix1, size=(y==1).sum())
    return (X, y)