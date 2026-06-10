# Lab 12 - Regression I

This lab covers nonparametric regression, especially Nadaraya-Watson kernel regression and smoothing splines.

## Regression Function

The regression target is usually written as:

```text
Y = g(X) + error
```

The goal is to estimate the unknown function `g(x)`.

## Nadaraya-Watson Kernel Regression

Nadaraya-Watson regression predicts at a point by taking a weighted average of nearby observed responses:

```text
g_hat(x) = sum_i K((x - x_i) / h) y_i / sum_i K((x - x_i) / h)
```

The bandwidth `h` controls locality. Small `h` gives a rough, low-bias, high-variance curve. Large `h` gives a smooth, high-bias, low-variance curve.

## Smoothing Splines

Smoothing splines fit a smooth global curve by balancing data fit against a roughness penalty. The smoothing parameter controls how much curvature is allowed.

## Local vs Global Methods

Nadaraya-Watson is local: a point mostly affects predictions nearby. Smoothing splines are more global because the fitted function is constrained as a whole.

## Error Types

Function estimation error compares `g_hat(x)` to the true `g(x)`, which is available only in simulations. Prediction error compares predictions to observed `y`, which is available in real datasets.

Choosing the smoothing parameter is a model selection problem and should usually be done with validation or cross-validation, not training error.
