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

## Key Formulas

Regression model:

```text
Y = g(X) + epsilon
```

Nadaraya-Watson estimator:

```text
g_hat_h(x) = sum_i K((x - x_i) / h) y_i
             / sum_i K((x - x_i) / h)
```

Gaussian kernel:

```text
K(u) = (1 / sqrt(2 pi)) exp(-u^2 / 2)
```

Function estimation error:

```text
MSE_g = (1 / n) sum_i (g(x_i) - g_hat(x_i))^2
```

Prediction error:

```text
MSE_y = (1 / n) sum_i (y_i - g_hat(x_i))^2
```

Smoothing spline objective:

```text
minimize sum_i (y_i - g(x_i))^2 + lambda integral (g''(t))^2 dt
```

## Hyperparameters and Failure Modes

For Nadaraya-Watson, the bandwidth `h` is the key hyperparameter. If `h` is too small, the estimate is noisy. If `h` is too large, important curvature is smoothed away.

For smoothing splines, `lambda` or the smoothing parameter controls roughness. Too little smoothing overfits noise. Too much smoothing underfits the true function.

As sample size grows, the optimal bandwidth usually decreases. Keeping `h` too large prevents the estimator from using the extra data to learn local structure.
