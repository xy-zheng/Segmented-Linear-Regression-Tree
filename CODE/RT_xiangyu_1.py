#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 11:30:43 2018

@author: zhengxiangyu
"""
import scipy
import scipy.optimize
import numpy
import copy
import sys
import time
import random
import math
from bisect import bisect_right
from operator import itemgetter

class Tree(object):
    def __init__(self, error, predict, stdev, start, num_points, index, region_info):
        self.error = error
        self.predict = predict
        self.stdev = stdev
        self.start =  start
        self.num_points = num_points
        self.region_info = region_info
        self.index = index[0]
        self.split_var = None
        self.split_val = None
        self.split_lab = None
        self.left = None
        self.right = None

    def lookup(self, x):
        """Returns the predicted value given the parameters."""
        if self.left == None:
            return self.predict
        if x[self.split_var] <= self.split_val:
            return self.left.lookup(x)
        return self.right.lookup(x)

    def lookup_sig2(self, x):
        if self.left == None:
            return self.stdev
        if x[self.split_var] <= self.split_val:
            return self.left.lookup_sig2(x)
        else:
            return self.right.lookup_sig2(x)


    def predict_all(self, data):
        """Returns the predicted values for some list of data points."""
        return map(lambda x: self.lookup(x), data)


    def find_weakest(self):
        """Finds the smallest value of alpha and
        the first branch for which the full tree
        does not minimize the error-complexity measure."""
        if (self.right == None):
            return float("Inf"), [self]
        b_error, num_nodes = self.get_cost_params()
        alpha = (self.error - b_error) / (num_nodes - 1)
        alpha_right, tree_right = self.right.find_weakest()
        alpha_left, tree_left = self.left.find_weakest()
        smallest_alpha = min(alpha, alpha_right, alpha_left)
        smallest_trees = []
        # if there are multiple weakest links collapse all of them
        if smallest_alpha == alpha:
            smallest_trees.append(self)
        if smallest_alpha == alpha_right:
            smallest_trees = smallest_trees + tree_right
        if smallest_alpha == alpha_left:
            smallest_trees = smallest_trees + tree_left
        return smallest_alpha, smallest_trees


    def prune_tree(self):
        """Finds {a1, ..., ak} and {T1, ..., Tk},
        the sequence of nested subtrees from which to
        choose the right sized tree."""
        trees = [copy.deepcopy(self)]
        alphas = [0]
        new_tree = copy.deepcopy(self)
        while 1:
            alpha, nodes = new_tree.find_weakest()
            for node in nodes:
                node.right = None
                node.left = None
            trees.append(copy.deepcopy(new_tree))
            alphas.append(alpha)
            # root node reached
            if (node.start == True):
                break
        return alphas, trees


    def get_cost_params(self):
        """Returns the branch error and number of nodes."""
        if self.right == None:
            return self.error, 1
        error, num_nodes = self.right.get_cost_params()
        left_error, left_num = self.left.get_cost_params()
        error += left_error
        num_nodes += left_num
        return error, num_nodes


    def get_length(self):
        """Returns the length of the tree."""
        if self.right == None:
            return 1
        right_len = self.right.get_length()
        left_len = self.left.get_length()
        return max(right_len, left_len) + 1

    def get_leaf_number(self):
        """Return the number of leaf nodes of the tree."""
        if self.right == None:
            return 1
        else:
            return self.left.get_leaf_number()+self.right.get_leaf_number()


def grow_tree(data, index, depth = 1, max_depth = 500, Nmin = 5, labels = [], start\
              = False, feat_bag = False, var_mi = 0.5, region_info = 'All Data'):
    """Function to grow a regression tree given some training data."""
    """parameters of Tree: error, predict, stdev, start, num_points, index. """
    root = Tree(region_error(data.values()), numpy.mean(numpy.array(data.values())),
        numpy.std(numpy.array(data.values())), start, len(data.values()), index, region_info)
    # regions has fewer than Nmin data points
    if (len(data.values()) <= Nmin):
        return root
    # length of tree exceeds max_depth
    if depth >= max_depth:
        return root
    num_vars = len(data.keys()[0])

    min_error = -1
    split_val = -1
    split_var = -1

    # Select variables to chose the split point from.
    # If feature bagging (for random forests) choose sqrt(p) variables
    # where p is the total number of variables.
    # Otherwise select all variables.
    if (feat_bag):
        cand_vars = random.sample(range(num_vars), int(num_vars**(var_mi)))
    else:
        cand_vars = range(num_vars)
    # iterate over parameter space, set() is to get the unique values
    for i in cand_vars:
        var_space = set([x[i] for x in data])
        # find optimal split point for parameter i
        for split in var_space:
            error = error_function(split_point = split, split_var = i, data = data, Nmin=Nmin)
            if error>0:
                if((error < min_error) or (min_error == -1)):
                    min_error = error
                    split_val = split
                    split_var = i
    # no more splits possible
    if split_var == -1:
        return root
    root.split_var = split_var
    root.split_val = split_val
    if len(labels)>split_var:
        root.split_lab = labels[split_var]
    else:
        root.split_lab = str(split_var)
    #print 'split lab is ', root.split_lab, 'split point is ', split_val
    data1 = {}
    data2 = {}
    for i in data:
        if i[split_var] <= split_val:
            data1[i] = data[i]
        else:
            data2[i] = data[i]
    #grow right and left branches
    # index plus one after each growing
    index[0] = index[0]+1
    region_info_left = root.split_lab + '<' + str(round(split_val,2))
    root.left = grow_tree(data1, index, depth + 1, max_depth = max_depth, Nmin = Nmin, labels = labels, feat_bag = feat_bag, region_info = region_info_left)
    index[0] = index[0]+1
    region_info_right = root.split_lab + '<' + str(round(split_val,2))
    root.right = grow_tree(data2, index, depth + 1, max_depth = max_depth, Nmin = Nmin, labels = labels, feat_bag = feat_bag, region_info = region_info_right)
    return root

def cvt(data, v, max_depth = 500, Nmin = 5, labels = {}):
    """Grows regression tree using v-fold cross validation.

    Data is a dictionary with elements of the form
    (x1, ..., xd) : y where x1, ..., xd are the parameter values and
     y is the response value.
     v is the number of folds for cross validation.
     max_depth is the maximum length of a branch emanating from the starting node.
     Nmin is the number of datapoints that must be present in a region to stop further partitions
     in that region.
     labels is a dictionary where the keys are the indices for the parameters in the data
     and the values are strings assigning a label to the parameters.
     See football_parserf.py for an example implementation."""
    full_tree = grow_tree(data, index = [1], depth = 1, max_depth = max_depth, Nmin = Nmin,
        labels = labels, start = True)
    full_a, full_t = full_tree.prune_tree()

    # ak' = (ak * ak+1)^(1/2)
    a_s = []
    for i in range(len(full_a) - 1):
        a_s.append((full_a[i] * full_a[i+1])**(.5))
    a_s.append(full_a[-1])
    # stratify data
    pairs = sorted(data.items(), key=itemgetter(1))

    # break the data into v subsamples of roughly equal size
    lv_s = [dict(pairs[i::v]) for i in range(v)]
    # list of tree sequences for each training set
    t_vs = []
    # list of testing data for each training set
    test_vs = []
    # list of alpha values for each training set
    alpha_vs = []

    # grow and prune each training set
    for i in range(len(lv_s)):
        train = {k: v for d in lv_s[:i] for (k, v) in d.items()}
        train.update({k: v for d in lv_s[(i + 1):] for (k, v) in d.items()})
        test = lv_s[i]
        full_tree_v = grow_tree(train,[1], 1, max_depth = max_depth, Nmin = Nmin,
            labels = labels, start = True)
        alphas_v, trees_v = full_tree_v.prune_tree()
        t_vs.append(trees_v)
        alpha_vs.append(alphas_v)
        test_vs.append(test)

    # choose the subtree that has the minimum cross-validated
    # error estimate
    min_R = float("Inf")
    min_ind = 0
    for i in range(len(full_t)):
        ak = a_s[i]
        R_k = 0
        for j in range(len(t_vs)):
            # closest alpha value in sequence v to
            # alphak'
            a_vs = alpha_vs[j]
            tr_vs = t_vs[j]
            alph_ind = bisect_right(a_vs, ak) - 1
            pairs = test_vs[j].items()
            para = [k[0] for k in pairs]
            va = [k[1] for k in pairs]
            pred_vals = tr_vs[alph_ind].predict_all(para)
            r_kv = numpy.sum((numpy.array(va) - numpy.array(pred_vals))**2)
            R_k = R_k + r_kv
        if (R_k < min_R):
            min_R = R_k
            min_ind = i
    return full_t[min_ind]



def error_function(split_point, split_var, data, Nmin):
    """Function to minimize when choosing split point."""
    data1 = []
    data2 = []
    for i in data:
        if i[split_var] <= split_point:
            data1.append(data[i])
        else:
            data2.append(data[i])
    if len(data1)<Nmin or len(data2)<Nmin:
        err = -1
    else:
        err = region_error(data1) + region_error(data2)
    return err

def region_error(data):
    """Calculates sum of squared error for some node in the regression tree."""
    data = numpy.array(data)
    return numpy.sum((data - numpy.mean(data))**2)

# return all the paths
def get_path(tree, li, path):
    """Get a list of all paths, where each element is one single path
    corresponding to a node.
    when call the function, the parameters shoulde be [] and [], for eg,
    res = get_paths(tree,[],[])."""

    if tree.right != None:
        li.append(tree.split_lab+'<'+str(tree.split_val))
        path=get_path(tree.left, li, path)
        li.pop()
        li.append(tree.split_lab+'>'+str(tree.split_val))
        path=get_path(tree.right, li, path)
        li.pop()
        return path
    else :
        path.append(copy.deepcopy(li))
        return path

# get the index set of all terminal/leaf nodes
def get_leaf_index(tree, leaves):
    """Get the index of all leaves. when function is called, leaves should be
    set as []
    see: get_leaf_index(tree, [])"""
    if tree.right != None:
        leaves = get_leaf_index(tree.left, leaves)
        leaves = get_leaf_index(tree.right, leaves)
        return leaves
    else:
        leaves.append(tree.index)
        return leaves

# get a dictionary of prediction beta all terminal/leaf nodes
def get_leaf_pred(tree, predicts):
    """key: leaf index ; value: beta
        the order is the same as leaves"""
    if tree.right != None:
        predicts = get_leaf_pred(tree.left, predicts)
        predicts = get_leaf_pred(tree.right, predicts)
        return predicts
    else:
        predicts.update({tree.index: tree.predict})
        return predicts

# get a dictionary of sigma (sigma = sse/(n-p-1)) over all terminal/leaf nodes
def get_leaf_sigma(tree, sigma):
    """key: leaf index ; value: sigma
        the order is the same as leaves"""
    if tree.right != None:
        sigma = get_leaf_sigma(tree.left, sigma)
        sigma = get_leaf_sigma(tree.right, sigma)
        return sigma
    else:
        sigma.update({tree.index: tree.stdev})
        return sigma

# return the leaf node where each of the record belong to
def get_matching_node(tree, x):
    """Return the index of the leaf node which the record x belongs to"""
    if tree.right != None:
        if x[tree.split_var] < tree.split_val:
            return get_matching_node(tree.left, x)
        else:
            return get_matching_node(tree.right, x)
    else:
        return tree.index

# partition the data in to all "leaves"
def data_partion(tree, data):
    """Return a large dictionary, where the keys are the index of leaves and
       for a certain leaf, the corresponding value is the sub-dictionary this leaf"""
    leaf_list = get_leaf_index(tree,[])
    data_leaf_index = numpy.array([get_matching_node(tree,i) for i in data.keys()])
    data_partion = {}
    for l in range(len(leaf_list)):
        sub_ind = list(numpy.where(data_leaf_index == leaf_list[l])[0])
        data_partion[leaf_list[l]] = {data.keys()[i]:data.values()[i] for i in sub_ind}
    return data_partion