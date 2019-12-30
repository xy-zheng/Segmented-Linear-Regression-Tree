#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 11:26:56 2019

@author: zhengxiangyu
"""
'--------------------------------import modules---------------------------------'
import sys
import numpy
import time
import pandas as pd
sys.path.append('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/3.share/')
import SLRT as lrt
import treeWriter_xiangyu as tw
import treeWriter_err_xiangyu as tw_err
import treeWriter_beta_xiangyu as tw_beta
'--------------------------------training data---------------------------------'
def num2str(x):
    if x==1:
        return 'a'
    else:
        if x==2:
            return 'b'
        else:
            return 'c'
numpy.random.seed(123)
x1 = numpy.random.random(1500)*20
x2 = numpy.random.random(1500)*20+5
x3 = numpy.random.random(1500)*10
x4 = numpy.random.randint(1,4,1500)
x_dis = [num2str(x) for x in x4]

x1_value=x1.copy()
x2_value=x2.copy()
x3_value=x3.copy()
x4_value=x4.copy()
y_1 = (3*x1_value)*(x2_value>15)+(-3)*x1_value*(x2_value<=15)+(-3*x2_value)*(x2_value>=10)-5*x2_value*(x2_value<10)+(x3_value)*(x1_value>10)\
    -x3*(x1_value<10)
y_2 = x3_value
y_2[numpy.where(x4_value==3)] = -3*x3_value[numpy.where(x4_value==3)]
y = y_1+y_2+numpy.random.randn(1500)

x_dis = [num2str(x) for x in x4]
x = zip(list(x1), list(x2), list(x3), list(x_dis))
data_train = pd.DataFrame(x,columns=['x1', 'x2', 'x3', 'x4'])
data_train['y']=y
'=========variables setting======='
var_all=numpy.array(['x1', 'x2', 'x3', 'x4', 'y'])
var_res=['y']
var_split=['x1', 'x2', 'x3', 'x4']
var_reg=['x1', 'x2', 'x3']
id_res = numpy.where(numpy.isin(var_all,var_res))[0]
id_cand_all = numpy.where(numpy.isin(var_all,var_split))[0]
id_value_all = numpy.where(numpy.isin(var_all,var_reg))[0]
id_dis = numpy.where(numpy.isin(var_all,['x4']))[0]
'===========lrt model=============='
start = time.clock()
model_lrt = lrt.grow_tree(data_train, id_value_all, id_cand_all, id_dis, id_res, index = [1],\
                          labels = var_all, Nmin = 50, start = True, Nsplit = 500)
end = time.clock()
print end-start


start = time.clock()
model_lrt_prune = lrt.cvt(data_train, id_value_all, id_cand_all, id_dis, id_res, v=10, labels=var_all,
                          Nmin=50, Nsplit=500)
end = time.clock()
print end-start
'=========result: plot tree========'
writer1 = tw.treeWriter(model_lrt)
writer1.write('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/3.share/examples/eg_simulation1/results/complete_tree.pdf')
# this tree plot contains the estimated standard error of epsilon, \hat{\sigma} at each leaf node
writer2 = tw_err.treeWriter(model_lrt)
writer2.write('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/3.share/examples/eg_simulation1/results/complete_tree_err.pdf')
# this tree plot contains the estimated linearregression function
writer3 = tw_beta.treeWriter(model_lrt,var_all)
writer3.write('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/3.share/examples/eg_simulation1/results/complete_tree_beta.pdf')

writer1 = tw.treeWriter(model_lrt_prune)
writer1.write('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/3.share/examples/eg_simulation1/results/pruned_tree.pdf')
# this tree plot contains the estimated standard error of epsilon, \hat{\sigma} at each leaf node
writer2 = tw_err.treeWriter(model_lrt_prune)
writer2.write('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/3.share/examples/eg_simulation1/results/pruned_tree_err.pdf')
# this tree plot contains the estimated linearregression function
writer3 = tw_beta.treeWriter(model_lrt_prune,var_all)
writer3.write('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/3.share/examples/eg_simulation1/results/pruned_tree_beta.pdf')
'=========result: print the selected splits for each leaf node========'
# suppose the tree contains L leaf nodes, then get_path function returns a list with L elements
# each element corresponds to the split rules for a leaf node
lrt.get_path(model_lrt,[],[])
'=========result: print the selected split levels for each candidate split variable========'
dic_splits={'x1': [], 'x2': [], 'x3': [], 'x4': []}
lrt.get_splits(model_lrt,dic_splits)
'=========result: leaf node indexes; parameters; standard errors========'
# print all leaf node indexes
lrt.get_leaf_index(model_lrt,[])
# print the linear parameters inside each leaf node
lrt.get_leaf_beta(model_lrt,{})
# print the estimated sigma inside each leaf node
lrt.get_leaf_sigma(model_lrt,{})
'=========result: to locate a data record (which leaf node it falls in) and the corresponding linear model========'
# take the first row as an example
data_record=data_train.iloc[0]
# the index of the leaf node it falls in
lrt.get_matching_node(model_lrt, data_record)
'=========result: to partition a dataframe by the split rules of the constructed tree========'
# lrt.data_partition returns a dataframe that an extra column "leaf" is added
data_train_partitioned = lrt.data_partition(model_lrt, data = data_train)