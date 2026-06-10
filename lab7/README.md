# Lab 7 - Logistic Regression Regularization

This lab covers regularized logistic regression in high-dimensional settings. Regularization adds a penalty to the loss function to control coefficient size and improve stability.

## Ridge, Lasso, and Elastic Net

**Ridge regression** uses an L2 penalty. It shrinks coefficients toward zero but usually does not set them exactly to zero. Ridge is useful when many variables have small effects or when predictors are correlated.

**Lasso regression** uses an L1 penalty. It can set coefficients exactly to zero, so it performs variable selection. Lasso is useful for sparse models, but it can behave unstably when predictors are highly correlated.

**Elastic net** combines L1 and L2 penalties. It can select variables while also keeping groups of correlated variables together better than pure lasso.

## Lambda and Coefficient Paths

The regularization strength is often denoted by lambda. Large lambda means strong shrinkage and simpler models. Small lambda means weaker shrinkage and more flexible models.

Coefficient path plots show how coefficients enter or leave the model as lambda changes.

## Prediction vs Variable Selection

The lambda selected by cross-validation is optimized for prediction, not necessarily for recovering the true relevant variables. A model can predict well while selecting too many irrelevant features or missing some true features.

## PSR and FDR

Positive Selection Rate measures how many truly relevant variables were selected. False Discovery Rate measures how many selected variables were actually irrelevant. Both are needed to evaluate feature selection quality.

## Key Formulas

Regularized logistic regression objective:

```text
minimize -l(beta) + penalty(beta)
```

Ridge penalty:

```text
penalty(beta) = lambda sum_j beta_j^2
```

Lasso penalty:

```text
penalty(beta) = lambda sum_j |beta_j|
```

Elastic net penalty:

```text
penalty(beta) = lambda [(1 - alpha) / 2 sum_j beta_j^2 + alpha sum_j |beta_j|]
```

Positive Selection Rate:

```text
PSR = |S_true intersect S_selected| / |S_true|
```

False Discovery Rate:

```text
FDR = |S_selected \\ S_true| / |S_selected|
```

## Hyperparameters and Failure Modes

`lambda` controls regularization strength. In scikit-learn, `C = 1 / lambda`, so smaller `C` means stronger regularization.

For elastic net, `alpha` or `l1_ratio` controls the mix between ridge and lasso. Values near 1 behave like lasso. Values near 0 behave like ridge.

If regularization is too weak, many noisy variables remain and FDR can be high. If it is too strong, true variables can be removed and PSR can be low.

Cross-validation may choose a model with good prediction but too many selected variables. For variable selection, a stronger penalty or a one-standard-error rule may be preferable.
