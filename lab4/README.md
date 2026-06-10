# Lab 4 - Logistic Regression

This lab covers logistic regression for binary classification. Logistic regression models the conditional probability of class 1 as:

```text
P(Y = 1 | X = x) = sigmoid(beta_0 + beta^T x)
```

The decision boundary is linear in the original feature space when a fixed threshold such as 0.5 is used.

## Maximum Likelihood

The coefficients are estimated by maximizing the log-likelihood of the observed binary labels. Equivalently, the model minimizes logistic loss.

## Linearly Separable Data

When classes are linearly separable, the unregularized maximum likelihood estimate may not be finite. The likelihood can keep improving as coefficient magnitudes grow, which leads to very large coefficients and unstable predictions.

## Regularization

L2 regularization, also called ridge regularization, penalizes large coefficients. It stabilizes estimation, especially under separation or multicollinearity. The fitted decision boundary may be similar, but probabilities and coefficient magnitudes become more controlled.

## Correct vs Misspecified Models

In simulation, fitting the full model should improve as sample size grows. If relevant variables are omitted, the model is misspecified and coefficient estimates can remain biased even with large datasets.

## Key Formulas

Sigmoid function:

```text
sigmoid(z) = 1 / (1 + exp(-z))
```

Logistic regression model:

```text
P(Y = 1 | X = x) = sigmoid(beta_0 + beta^T x)
```

Log-odds interpretation:

```text
log(P(Y = 1 | X = x) / P(Y = 0 | X = x)) = beta_0 + beta^T x
```

Log-likelihood:

```text
l(beta) = sum_i y_i log(p_i) + (1 - y_i) log(1 - p_i)
```

L2-regularized objective:

```text
maximize l(beta) - lambda sum_j beta_j^2
```

## Hyperparameters and Failure Modes

The main regularization hyperparameter is `lambda`, or equivalently `C = 1 / lambda` in scikit-learn. Large `C` means weak regularization. Small `C` means strong regularization.

If regularization is too weak, coefficients can become very large under separation and probabilities can become overconfident. If regularization is too strong, coefficients shrink too much and the model underfits.

Feature scaling matters when regularization is used. Without scaling, variables measured on larger numeric scales may be penalized differently.
