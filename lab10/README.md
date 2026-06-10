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
