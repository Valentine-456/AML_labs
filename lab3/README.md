# Lab 3 - Density Estimation

This lab covers nonparametric density estimation, mainly kernel density estimation (KDE). Density estimation tries to approximate an unknown probability density function from observed samples without assuming a fixed parametric form such as a Gaussian distribution.

## Kernel Density Estimation

KDE estimates a density by placing a small smooth kernel around every observation and averaging the kernels:

```text
f_hat(x) = (1 / (n h)) sum_i K((x - x_i) / h)
```

`K` is the kernel function and `h` is the bandwidth.

## Kernel and Bandwidth Effects

The kernel controls the local shape of each contribution. Common choices include Gaussian, Epanechnikov, and tophat kernels. In practice, the bandwidth usually matters more than the exact kernel.

A small bandwidth gives a rough estimate with low bias and high variance. A large bandwidth gives a smooth estimate with high bias and low variance. Bandwidth selection is therefore a bias-variance tradeoff.

## KDE Resampling

Sampling from a KDE can be done by selecting one observed sample point and adding kernel noise. A KDE fitted to a large artificial sample generated this way should be close to the original KDE because both represent the same smoothed empirical distribution.

## Density Estimation in Classification

Density estimates can be used inside Naive Bayes classifiers by estimating `P(X_j | Y)` for each feature and class. KDE Naive Bayes is more flexible than Gaussian Naive Bayes, but still assumes conditional feature independence.

## Key Formulas

Kernel density estimator:

```text
f_hat_h(x) = (1 / (n h)) sum_i K((x - x_i) / h)
```

Gaussian kernel:

```text
K(u) = (1 / sqrt(2 pi)) exp(-u^2 / 2)
```

Mean squared error on evaluation points:

```text
MSE = (1 / K) sum_j (f(x_j) - f_hat(x_j))^2
```

KDE Naive Bayes classifier:

```text
y_hat = argmax_k log P(Y = k) + sum_j log f_hat_{j,k}(x_j)
```

## Hyperparameters and Failure Modes

The bandwidth `h` is the main hyperparameter. If `h` is too small, the density estimate becomes spiky and overfits noise. If `h` is too large, the estimate becomes overly smooth and can merge separate modes.

Kernel choice matters less than bandwidth, but compact kernels such as tophat or Epanechnikov can behave differently near sparse regions.

In KDE Naive Bayes, using too small a bandwidth can assign almost zero density to valid test points. Using too large a bandwidth can remove useful class-specific structure.
