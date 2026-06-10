# Lab 14 - Multi-Label Classification

This lab covers multi-label classification, where each observation can belong to several classes at the same time. The response is a binary vector:

```text
Y = (Y_1, ..., Y_K)
```

Each component indicates whether one label is active.

## Binary Relevance

Binary Relevance trains one independent binary classifier per label. It is simple and scalable, but it assumes labels are conditionally independent given the features.

## Classifier Chains

Classifier Chains train label classifiers sequentially. Later classifiers receive the original features plus earlier predicted labels as inputs.

This allows the model to use label dependencies, but performance can depend strongly on the label order. Errors made early in the chain can propagate.

## Ensemble of Classifier Chains

An ensemble of classifier chains trains many chains with different label orders and combines their predictions. This reduces dependence on a single arbitrary order and is usually more stable than one chain.

## Circular Chain Classifier

The circular chain idea fits one conditional model for each label using all other labels as additional predictors. During prediction, unknown labels are updated iteratively until the vector stabilizes.

This can be viewed as searching for a self-consistent label vector. It is heuristic and does not necessarily define one coherent joint probability model.

## Evaluation Metrics

Subset accuracy requires the entire predicted label vector to be correct, so it is strict.

Hamming score evaluates labels independently and is less strict.

Jaccard score focuses on overlap between true and predicted active labels and is useful when active labels are rare.
