# Lab 5 - Evaluation Methods

This lab covers how to estimate classification performance and how evaluation choices affect conclusions. The main target is the generalization error: the expected error on new unseen data.

## Evaluation Schemes

**Refitting error** trains and tests on the same data. It is usually optimistic because the model has already seen the examples.

**K-fold cross-validation** splits the data into folds, repeatedly trains on `K-1` folds, and tests on the held-out fold. It gives a more realistic estimate of generalization.

**Bootstrap out-of-bag error** repeatedly trains on bootstrap samples and tests on observations not selected in each bootstrap sample.

**Bootstrap 0.632 error** combines apparent training error and out-of-bag error. It is designed to correct some pessimism of pure bootstrap testing while avoiding the strong optimism of refitting.

## ROC vs Precision-Recall

ROC curves show the tradeoff between true positive rate and false positive rate. They are useful for threshold-independent evaluation.

Precision-recall curves focus on the positive class. They are often more informative when classes are imbalanced or when the positive class is the main class of interest.

## Threshold Choice

Accuracy depends on the classification threshold. A threshold of 0.5 is natural when probabilities are calibrated and class costs are equal. Balanced accuracy gives equal weight to both classes, so it is more appropriate under class imbalance.

If false positives and false negatives have different costs, the threshold should be shifted toward reducing the more expensive mistake.

## Key Formulas

Accuracy:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Balanced accuracy:

```text
Balanced accuracy = 0.5 * (TP / (TP + FN) + TN / (TN + FP))
```

Precision and recall:

```text
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

Bootstrap 0.632 estimate:

```text
Err_0.632 = 0.368 * Err_train + 0.632 * Err_bootstrap
```

Threshold classification rule:

```text
y_hat = 1 if P(Y = 1 | X = x) > t, else 0
```

## Hyperparameters and Failure Modes

For cross-validation, the number of folds controls the bias-variance tradeoff of the estimate. Too few folds can be pessimistic because training sets are smaller. Too many folds can be computationally expensive and high variance.

For bootstrap, the number of bootstrap repetitions controls Monte Carlo stability. Too few repetitions can give noisy estimates.

Accuracy can be misleading under class imbalance. ROC AUC can also look good when the positive class is rare, while precision-recall curves may reveal poor positive-class precision.
