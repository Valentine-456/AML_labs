# Lab 1-2 - Bayesian Classifiers

This lab covers classical generative classification methods. The central idea is to model the class-conditional distribution `P(X | Y)` and combine it with class priors `P(Y)` using Bayes' rule:

```text
P(Y = k | X = x) proportional to P(X = x | Y = k) P(Y = k)
```

The predicted class is the one with the largest posterior probability.

## Methods Covered

**Naive Bayes** assumes that features are conditionally independent given the class. This makes estimation simple and stable, especially with small datasets, but the independence assumption is often unrealistic.

**Linear Discriminant Analysis (LDA)** assumes each class follows a multivariate Gaussian distribution with a shared covariance matrix. This produces linear decision boundaries. LDA is useful when classes are approximately Gaussian and have similar covariance structure.

**Quadratic Discriminant Analysis (QDA)** assumes each class follows a multivariate Gaussian distribution with its own covariance matrix. This produces quadratic decision boundaries and is more flexible than LDA, but it needs more data to estimate covariance matrices reliably.

## Main Differences

Naive Bayes is the simplest and most restrictive because it ignores feature dependence. LDA models dependence through one shared covariance matrix and gives linear boundaries. QDA models class-specific covariance matrices and gives curved boundaries, but it is more prone to overfitting.

## Practical Notes

Use Naive Bayes when data are limited or high-dimensional. Use LDA when linear separation is plausible and covariance structures are similar. Use QDA when classes have visibly different covariance patterns and enough observations are available.

## Key Formulas

Bayes rule:

```text
P(Y = k | X = x) = P(X = x | Y = k) P(Y = k) / P(X = x)
```

Naive Bayes decision rule:

```text
y_hat = argmax_k P(Y = k) product_j P(X_j = x_j | Y = k)
```

LDA discriminant function:

```text
delta_k(x) = x^T Sigma^(-1) mu_k
             - 0.5 mu_k^T Sigma^(-1) mu_k
             + log pi_k
```

QDA discriminant function:

```text
delta_k(x) = -0.5 log |Sigma_k|
             - 0.5 (x - mu_k)^T Sigma_k^(-1) (x - mu_k)
             + log pi_k
```

## Hyperparameters and Failure Modes

Naive Bayes usually has few hyperparameters, but smoothing may be needed for discrete features. Without smoothing, unseen feature values can produce zero likelihood and dominate the posterior.

LDA has no main tuning parameter in its basic form, but it depends strongly on the shared covariance estimate. If covariance matrices differ strongly between classes, LDA can underfit.

QDA estimates one covariance matrix per class. If the sample size is small compared with the number of features, covariance estimates can become unstable or singular.
