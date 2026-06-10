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
