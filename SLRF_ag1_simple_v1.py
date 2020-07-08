#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 21:49:44 2018

@author: zhengxiangyu
"""


import sys
import numpy
sys.path.append('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/5.Code_Examples/CODE_python_3.7')
import SLRT_ag1_simple_v1 as lrt

class Forest(object):
    def __init__(self, trees):
        self.trees = trees # trees is a list of all trees
    
    def lookup(self, x):
        """Returns the predicted value given the parameters."""
        preds = list(map(lambda t: t.lookup(x), self.trees))
        return numpy.mean(preds)
    
    def predict_all(self, data):
        """Returns the predicted values for a list of data points."""
        return data.apply(lambda x:self.lookup(x), axis = 1)
    

    def lookup_wt(self, x):
        sig = 1.0/numpy.array(list(map(lambda t: t.lookup_sig2(x), self.trees)))
        sum = numpy.sum(sig)
        weight = numpy.array([i/sum for i in sig])
        preds_raw = list(map(lambda t: t.lookup(x), self.trees))
        preds = preds_raw * weight
        return numpy.sum(preds)

    def weighted_predict_all(self, data):
        return data.apply(lambda x:self.lookup_wt(x), axis = 1)

def make_forest(data, B, id_value_all, id_cand_all, id_dis, id_res, labels, max_depth = 500, Nmin = 100, Nsplit = 10, sNsplit = 5, mi=0.5):
    """Function to grow a random forest given some training data."""
    trees = []
    for b in range(B):
        boot = data.sample(frac = 1, replace = True)
        trees.append(lrt.grow_tree(boot, id_value_all, id_cand_all, id_dis, id_res, index = [1], \
                                   labels = labels, Nmin = Nmin, sNsplit = sNsplit, max_depth = max_depth, \
                              start = True, feat_bag = True, region_info = 'All Data',mi=mi))
    return Forest(trees)


from concurrent.futures import ProcessPoolExecutor
import time
def make_forest_parallel(data, B, id_value_all, id_cand_all, id_dis, id_res, labels, max_depth = 20, Nmin = 100, Nsplit = 10, sNsplit = 5, mi=0.5):
    """Function to grow a random forest given some training data."""
    trees = []
    def single_tree(boot):
        return(lrt.grow_tree(boot, id_value_all, id_cand_all, id_dis, id_res, index = [1], \
                             labels = labels, Nmin = Nmin, sNsplit = sNsplit, max_depth = max_depth, \
                             start = True, feat_bag = True, region_info = 'All Data',mi=mi))
    bootstrap_samples = []
    for b in range(B):
        boot = data.sample(frac = 1, replace = True)
        bootstrap_samples.append(boot)
    time1 = time.time()
    single_tree(bootstrap_samples[0])
    time2 = time.time()
    print(time2-time1,'\n')
    # max_workers=2
    with ProcessPoolExecutor() as pool:
        trees = list(pool.map(single_tree, bootstrap_samples))
    return Forest(trees)


