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
