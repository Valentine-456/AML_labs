# Lab 13 - Semi-Supervised Learning

This lab covers semi-supervised learning, where only some training observations have labels and the remaining observations are unlabeled.

## Core Idea

Semi-supervised methods try to use both labeled and unlabeled data. The unlabeled data do not directly provide class labels, but they can reveal the geometry, clusters, or manifold structure of the feature distribution.

## Naive Supervised Baseline

The naive method trains a standard classifier using only the labeled examples. It ignores unlabeled data. This is simple and reliable when enough labels are available, but it can perform poorly when the number of labels is very small.

## Self-Training

Self-training first fits a classifier on labeled data, then assigns pseudo-labels to high-confidence unlabeled observations. The classifier is retrained using the expanded labeled set.

The risk is confirmation bias: early wrong pseudo-labels can reinforce later mistakes.

## Label Propagation

Label propagation builds a graph over observations and spreads labels through the graph. Nearby or strongly connected observations are encouraged to have similar labels.

It works best when the data geometry matches the class structure.

## Label Spreading

Label spreading is similar to label propagation but adds regularization, making it more robust to noise. It usually changes labels more smoothly and can be more stable.

## Effect of Number of Labels

With very few labeled examples, semi-supervised methods can outperform purely supervised training. As the number of labeled examples increases, the advantage of using unlabeled data often decreases.

## Key Formulas

Semi-supervised training labels:

```text
y_i is known for labeled points
y_i = -1 or unknown for unlabeled points
```

Self-training pseudo-label rule:

```text
assign y_hat_i if max_k P(Y = k | x_i) >= threshold
```

Graph similarity with RBF weights:

```text
w_ij = exp(-gamma ||x_i - x_j||^2)
```

F1 score:

```text
F1 = 2 * precision * recall / (precision + recall)
```

ROC AUC:

```text
AUC = probability that a random positive example receives a higher score
      than a random negative example
```

## Hyperparameters and Failure Modes

For self-training, the confidence threshold controls which pseudo-labels are accepted. If the threshold is too low, many wrong pseudo-labels can be added. If it is too high, too few unlabeled points are used.

For label propagation and label spreading, `gamma`, kernel choice, and graph connectivity control how labels move through the dataset. If gamma is too small, the graph is too dense and labels oversmooth. If gamma is too large, the graph becomes too local and labels may not propagate.

For label spreading, `alpha` controls clamping vs smoothing. Strong smoothing can wash out true label boundaries.

Semi-supervised methods rely on the cluster or manifold assumption: nearby points should have similar labels. If this assumption is false, unlabeled data can hurt performance.
