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

## Key Formulas

Multi-label target:

```text
Y_i = (Y_i1, ..., Y_iK), where Y_ik in {0, 1}
```

Binary Relevance:

```text
fit K models: P(Y_k = 1 | X), for k = 1, ..., K
```

Classifier Chain factorization for one order:

```text
P(Y | X) = product_k P(Y_k | X, Y_1, ..., Y_{k-1})
```

Subset accuracy:

```text
Subset accuracy = (1 / n) sum_i I(Y_i = Y_hat_i)
```

Hamming score:

```text
Hamming score = (1 / (n K)) sum_i sum_k I(Y_ik = Y_hat_ik)
```

Jaccard score:

```text
Jaccard_i = |active(Y_i) intersect active(Y_hat_i)|
            / |active(Y_i) union active(Y_hat_i)|
```

## Hyperparameters and Failure Modes

For Binary Relevance, the main hyperparameters belong to the base classifier. The main modeling risk is ignoring label dependence.

For Classifier Chains, label order is a major hyperparameter. A poor order can propagate errors from early labels to later labels.

For Ensemble Classifier Chains, the number of chains controls stability. Too few chains may still depend strongly on random orders. More chains improve stability but increase computation.

For Circular Chain Classifiers, initialization, update order, and maximum iterations matter. The method may fail to converge or may converge to different label vectors depending on initialization.
