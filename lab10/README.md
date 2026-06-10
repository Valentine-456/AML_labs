# Lab 10 - Feature Selection

This lab covers feature ranking and feature selection in problems where only some variables are truly relevant. The goal is to identify useful predictors and understand how selection affects prediction.

## Nonlinear Signal

The generated datasets use nonlinear rules based on squared values or absolute values of the relevant variables. Because the relationship is symmetric, simple marginal correlation with the response may be weak even for important variables.

## Random Forest Importance

Random forest importance can be measured in two common ways.

Mean decrease in impurity measures how much splits on a feature reduce tree impurity. It is fast but can be biased toward variables with many possible split points.

Permutation importance measures how much model performance decreases when a feature is randomly shuffled. It is usually more directly tied to predictive value, but it is more computationally expensive.

## Boruta

Boruta is an all-relevant feature selection method. It creates shadow variables by shuffling original features, trains a random forest, and compares real feature importance against shadow feature importance.

The shadow variables provide a reference distribution for noise. Comparing against the best shadow variable is more meaningful than checking whether importance is merely positive.

## Prediction After Selection

Selecting too few features can miss signal and increase bias. Selecting too many features can add noise and increase variance. The best number of selected features depends on sample size, signal strength, and model type.

## Key Formulas

Dataset 1 signal rule:

```text
Y = 1 if sum_{j=1}^k X_j^2 > median(ChiSquare_k)
```

Dataset 2 signal rule:

```text
Y = 1 if sum_{j=1}^k |X_j| > k
```

Permutation importance:

```text
Importance_j = Score(model, X, y) - Score(model, X with feature j shuffled, y)
```

Feature recovery probability:

```text
P(success) ~= number of successful simulations / number of simulations
```

## Hyperparameters and Failure Modes

Important random forest hyperparameters for feature importance include `n_estimators`, `max_features`, `max_depth`, and `min_samples_leaf`.

Too few trees make feature rankings noisy. Very deep trees can overfit and inflate importance for noisy variables. Correlated features can split importance between each other, making individual rankings harder to interpret.

Permutation importance depends on the evaluation set. If the validation set is small, importance estimates can be noisy. If features are strongly correlated, shuffling one feature may underestimate its true importance because correlated features still carry similar information.

For Boruta, the number of random forest runs and shadow-variable comparison rule affect stability. Too few repetitions can produce unstable selected sets.
