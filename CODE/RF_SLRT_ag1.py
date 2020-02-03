#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 21:49:44 2018

@author: zhengxiangyu
"""


import sys
import numpy
sys.path.append('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/5.Code_Examples/CODE/')
import SLRT_ag1_simple as lrt

class Forest(object):
    def __init__(self, trees):
		self.trees = trees # trees is a list of all trees

    def lookup(self, x):
		"""Returns the predicted value given the parameters."""
		preds = map(lambda t: t.lookup(x), self.trees)
		return numpy.mean(preds)

    def predict_all(self, data):
		"""Returns the predicted values for a list of data points."""
		return data.apply(lambda x:self.lookup(x), axis = 1)

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

    def predict_all_weighted(self, data):
        return data.apply(lambda x:self.lookup_wt(x), axis = 1)


def make_forest(data, B, id_value_all, id_cand_all, id_dis, id_res, labels, max_depth = 500, Nmin = 100, sNsplit = 5, mi=0.5):
    """Function to grow a random forest given some training data."""
    trees = []
    for b in range(B):
        boot = data.sample(frac = 1, replace = True)
        trees.append(lrt.grow_tree(boot, id_value_all, id_cand_all, id_dis, id_res, index = [1], labels = labels, max_depth = max_depth, Nmin = Nmin, sNsplit = sNsplit, \
                              start = True, feat_bag = True, region_info = 'All Data',mi=mi))
    return Forest(trees)