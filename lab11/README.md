# Lab 11 - Support Vector Machines

This lab covers linear and kernel Support Vector Machines (SVMs). SVMs are margin-based classifiers: they choose a decision boundary that separates classes while maximizing the margin between classes.

## Linear SVM

A linear SVM finds a hyperplane that separates the classes. The observations that lie on or inside the margin are called support vectors. These observations determine the fitted boundary.

Support vectors are not necessarily misclassified. They are the points closest to the boundary or violating the margin.

## Logistic Regression vs Linear SVM

Both methods can produce linear decision boundaries, but they optimize different objectives. Logistic regression maximizes likelihood and uses all observations through the logistic loss. Linear SVM optimizes a hinge-loss margin objective and depends mainly on observations near the boundary.

Logistic regression directly estimates probabilities. A standard SVM does not estimate posterior probabilities unless an additional calibration step is used.

## The C Parameter

`C` controls the penalty for margin violations. Small `C` allows a wider margin with more violations, giving stronger regularization. Large `C` penalizes violations heavily and can produce a narrower margin with fewer training errors.

## Kernel SVM

Kernel SVMs apply the SVM idea in an implicit feature space. Polynomial kernels model polynomial boundaries. RBF kernels produce flexible local decision boundaries.

For the RBF kernel, small gamma gives very smooth boundaries. Large gamma gives highly local boundaries and can overfit.

## Key Formulas

Linear decision function:

```text
f(x) = w^T x + b
```

Hard-margin idea:

```text
minimize 0.5 ||w||^2
subject to y_i (w^T x_i + b) >= 1
```

Soft-margin objective:

```text
minimize 0.5 ||w||^2 + C sum_i xi_i
subject to y_i (w^T x_i + b) >= 1 - xi_i
```

Hinge loss:

```text
max(0, 1 - y_i f(x_i))
```

RBF kernel:

```text
K(x, z) = exp(-gamma ||x - z||^2)
```

Polynomial kernel:

```text
K(x, z) = (gamma x^T z + coef0)^degree
```

## Hyperparameters and Failure Modes

`C` controls the penalty for margin violations. If `C` is too small, the model allows many violations and may underfit. If `C` is too large, the model tries too hard to classify training points correctly and may overfit.

For RBF SVM, `gamma` controls locality. If `gamma` is too small, the boundary is too smooth and underfits. If `gamma` is too large, each point has very local influence and the boundary can become noisy.

For polynomial kernels, `degree`, `gamma`, and `coef0` control boundary complexity. High degree can overfit and become numerically unstable.

SVMs are sensitive to feature scaling because margins and distances depend on feature units.
