#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 21:49:44 2018

@author: zhengxiangyu
"""


import sys
sys.path.append('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/0.Module')
import numpy
import RT_xiangyu_1 as rt1

class Forest(object):
    def __init__(self, trees):
		self.trees = trees # trees is a list of all trees

    def lookup(self, x):
        sig = 1.0/numpy.array(map(lambda t: t.lookup_sig2(x), self.trees))
        ones_array = numpy.ones(sig.shape[0])
        sig_mean = numpy.mean(sig)
        ones_array[(sig/sig_mean)>1000]=0
        ones_array[(sig/sig_mean)<0.001]=0
        unweight = ones_array/numpy.sum(ones_array)
        preds_raw = map(lambda t: t.lookup(x), self.trees)
        preds = preds_raw * unweight
        return numpy.sum(preds)

    def lookup_wt(self, x):
        sig = 1.0/numpy.array(map(lambda t: t.lookup_sig2(x), self.trees))
        # delete the extrem values
        sig_mean=numpy.mean(sig)
        sig[(sig/sig_mean)>1000]=0
        sig[(sig/sig_mean)<0.001]=0
        # calculate weight that is based on the one without extreme values
        sig_sum = numpy.sum(sig)
        weight = numpy.array([i/sig_sum for i in sig])
        preds_raw = map(lambda t: t.lookup(x), self.trees)
        preds = preds_raw * weight
        return numpy.sum(preds)

    def predict_all(self, data):
		"""Returns the predicted values for a list of data points."""
		return map(lambda x: self.lookup(x), data)

    def predict_all_weighted(self, data):
		"""Returns the predicted values for a list of data points."""
		return map(lambda x: self.lookup_wt(x), data)
def make_boot(pairs, n):
	"""Construct a bootstrap sample from the data."""
	inds = numpy.random.choice(n, size=n, replace=True)
	return dict(map(lambda x: pairs[x], inds))

def make_forest(data, B, var_mi = 0.5, max_depth = 500, Nmin = 100, labels = []):
	"""Function to grow a random forest given some training data."""
	trees = []
	n = len(data)
	pairs = data.items()
	for b in range(B):
		boot = make_boot(pairs, n)
		trees.append(rt1.grow_tree(boot, index = [1], Nmin = Nmin, max_depth = max_depth, labels = labels, start = True, feat_bag = True, var_mi = var_mi))
	return Forest(trees)