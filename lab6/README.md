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
