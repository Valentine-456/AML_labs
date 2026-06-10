# Labs 8-9 - Ensemble Methods

These labs cover ensemble classifiers: methods that combine many base models to obtain better predictive performance than a single model.

## Bagging

Bagging trains base learners on bootstrap samples and averages their predictions. For classification, averaging is usually done by majority vote. Bagging mainly reduces variance and works especially well with unstable learners such as decision trees.

## AdaBoost

AdaBoost trains weak classifiers sequentially. After each iteration, misclassified observations receive more influence, so later classifiers focus on harder cases. Each classifier receives a voting weight based on its weighted error.

Weak learners with smaller error get larger voting weights. If a learner has error 0, the weight becomes extremely large. If its error is at least 0.5 in binary classification, it is no better than random guessing and should not be added in the usual AdaBoost formulation.

## Gradient Boosting

Gradient boosting also builds models sequentially, but it fits new learners to the residual errors or negative gradients of a loss function. It is a general framework that can optimize many losses.

## Random Forest

Random forest averages many randomized trees. Compared with bagging, it samples candidate features at each split, reducing correlation between trees and improving the ensemble average.

## XGBoost

XGBoost is an optimized gradient boosting implementation with regularization, shrinkage, tree constraints, and efficient training. It is often strong in tabular prediction tasks.

## Bias and Variance

Bagging and random forests mostly reduce variance. Boosting can reduce bias and variance, but can overfit if the number of iterations is too large or regularization is too weak.

## Key Formulas

AdaBoost weighted error:

```text
err_m = sum_i w_i I(y_i != f_m(x_i))
```

AdaBoost classifier weight:

```text
alpha_m = log((1 - err_m) / err_m)
```

AdaBoost prediction:

```text
y_hat = sign(sum_m alpha_m f_m(x))
```

Gradient boosting additive model:

```text
F_M(x) = F_0(x) + sum_m nu h_m(x)
```

Random forest prediction:

```text
y_hat = majority_vote(tree_1(x), ..., tree_B(x))
```

## Hyperparameters and Failure Modes

For AdaBoost, important hyperparameters are `n_estimators`, base tree depth, and learning rate if shrinkage is used. Too many iterations or too deep base trees can overfit.

For gradient boosting and XGBoost, important hyperparameters include `n_estimators`, `learning_rate`, `max_depth`, `subsample`, and regularization terms. A large learning rate can overfit quickly. A very small learning rate needs many trees.

For random forests, important hyperparameters include `n_estimators`, `max_features`, `max_depth`, and `min_samples_leaf`. Too few trees gives unstable predictions. Trees that are too shallow can underfit.

If AdaBoost gets a base learner with error 0, its vote weight tends to infinity. If the error is at least 0.5, the learner is not useful in the binary AdaBoost update.
