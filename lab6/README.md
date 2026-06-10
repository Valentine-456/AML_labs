# Lab 6 - Classification Trees

This lab covers decision trees, pruning, bagging, and random forests. Tree-based classifiers split the feature space into rectangular regions and assign a class prediction in each region.

## Classification Trees

A decision tree recursively chooses feature splits that reduce impurity. Common impurity criteria include Gini impurity and entropy. Tree depth, minimum split size, and splitter strategy affect model complexity.

Deep trees have low bias but high variance. They can fit complex patterns, but they are sensitive to small data changes.

## Cost-Complexity Pruning

Pruning removes branches that add little predictive value. Cost-complexity pruning balances training fit against tree size using a penalty parameter. A pruned tree is usually easier to interpret and may generalize better.

## Bagging

Bagging trains many trees on bootstrap samples and combines predictions by voting. It mainly reduces variance. Since individual trees are unstable, averaging many trees often improves predictive performance.

## Random Forest

Random forest extends bagging by also randomizing the features considered at each split. This decorrelates trees, making the average more effective. Random forests are usually stronger predictors than bagged trees, but less interpretable than a single tree.

## Main Differences

A single tree is interpretable but unstable. Bagging stabilizes trees through averaging. Random forest usually improves further by reducing correlation between trees.

## Key Formulas

Gini impurity:

```text
Gini(node) = 1 - sum_k p_k^2
```

Entropy:

```text
Entropy(node) = - sum_k p_k log(p_k)
```

Information gain:

```text
Gain = Impurity(parent)
       - sum_children (n_child / n_parent) Impurity(child)
```

Cost-complexity pruning objective:

```text
R_alpha(T) = R(T) + alpha * |T|
```

Bagging prediction:

```text
y_hat = majority_vote(f_1(x), ..., f_B(x))
```

## Hyperparameters and Failure Modes

Important tree hyperparameters include `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`, and `ccp_alpha`.

If a tree is too deep, it can overfit noise. If it is too shallow, it underfits. If `min_samples_leaf` is too small, terminal regions can be unstable. If `ccp_alpha` is too large, pruning removes useful structure.

For bagging and random forests, `n_estimators` controls the number of trees. Too few trees gives unstable ensemble estimates. Very many trees improve stability but increase computation.

For random forests, `max_features` controls tree decorrelation. Too large makes trees similar to bagging. Too small can make individual trees weak.
