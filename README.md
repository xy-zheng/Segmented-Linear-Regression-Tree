# Segmented-Linear-Regression-Tree

This repository includes the Python code for SLRT(Segmented Linear Regression Tree) in our paper "Partitioning Structure Learning for Segmented Linear Regression Trees" (NeurIPS 2019).

* SLRT.py

The SLRT module is the implementation of Algorithm 1 in section 3.1.
Please refer to 'eg_simulation1.py' for an example of the application, which is also the simulation example in section 5.1.

* SLRT_faster.py

The SLRT_faster module is a faster implementation of Algorithm 1, where we firstly choose the split variable by investigating a small number of split levels.
Please refer to the folder 'eg_Boston_housing' for examples of applications.
 
* RT.py 
 
 The RT module is the implementations of CART, see "Classification and Regression Trees" by Breiman et al. (1984).

* tree_lasso.py

This module were called in the examples on public datasets (in folders 'examples/eg_xxx'), which combines the data partitioning results with LASSO estimation on leaves.

* treeWriter.py, treeWriter_err.py and treeWriter_beta.py

The three modules are all for plotting the tree.
treeWriter_err.py is for tree plot containing the estimated standard error at each leaf node.
treeWriter_beta.py is for tree plot containing the estimated linearregression function.
Please refer to 'examples/eg_simulation1' for the use of these modules. The figure 1 in our paper was the output of treeWriter.py.
