# Segmented-Linear-Regression-Tree

This repository includes the Python code for SLRT(Segmented Linear Regression Tree) in our paper "Partitioning Structure Learning for Segmented Linear Regression Trees" (NeurIPS 2019).

* SLRT_alg1_simple.py

The SLRT module is for Algorithm 1 for conditionally uncorrelated regressors.
The tree construction uses the simple stopping rule and includes the function for pruning (cvt()).

* SLRT_alg1_testing.py

The SLRT module is for Algorithm 1 for conditionally uncorrelated regressors.
The tree construction uses the hypothesis testing based stopping rule.

* SLRT_alg2_simple.py

The SLRT module is for Algorithm 2 for correlated regressors.
The tree construction uses the simple stopping rule and includes the function for pruning (cvt()).

* SLRT_alg2_testing.py

The SLRT module is for Algorithm 2 for correlated regressors.
The tree construction uses the hypothesis testing based stopping rule.

* tree_lasso.py

This module were called in the examples on public datasets (in folders 'examples/eg_xxx'), which combines the data partitioning results with LASSO estimation on leaves.

* plot_tree.py

This is for plotting the tree structures.

* example_illustration.py

To illustrate the parameters setting for SLRT algorithm, where the variables for regression and for splitting can be assigned by users.
