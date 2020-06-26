#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhengxiangyu
"""
import sys
import numpy
import pandas as pd
import math
from sklearn import tree
from IPython.display import Image
sys.path.append('/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/5.Code_Examples/CODE_python_3.7/')
import SLRT_ag1_simple_py3 as lrt_simple
import SLRT_ag1_testing_py3 as lrt_testing
import treeWriter_xiangyu_0626 as tw
from sklearn.datasets import load_boston
def num2str(x):
    dict_x4={1:'a',2:'b',3:'c'}
    return dict_x4[x]
# require x and y are numpy.array
def rmse(x, y):
    return numpy.sqrt(((x-y)**2).mean())
'=============================1. Generate Datasets============================='             
# 1.1 generate training data
numpy.random.seed(123)
x1 = numpy.random.random(1500)*20
x2 = numpy.random.random(1500)*20+5
x3 = numpy.random.random(1500)*10
x4 = numpy.random.randint(1,4,1500)

data_train=pd.DataFrame()
data_train['x1']=x1
data_train['x2']=x2
data_train['x3']=x3
data_train['x4']=[num2str(x) for x in x4]

y_1 = (3*x1)*(x2>15)+(-3)*x1*(x2<=15)+(-3*x2)*(x2>=10)-5*x2*(x2<10)+(x3)*(x1>10)\
    -x3*(x1<10)
y_2 = x3.copy()
y_2[numpy.where(x4==3)] = -3*x3[numpy.where(x4==3)]
y = y_1+y_2+numpy.random.randn(1500)

data_train['y']=y
#1.2   generate testing data
numpy.random.seed(456)
x1 = numpy.random.random(500)*20
x2 = numpy.random.random(500)*20+5
x3 = numpy.random.random(500)*10
x4 = numpy.random.randint(1,4,500)

data_test=pd.DataFrame()
data_test['x1']=x1
data_test['x2']=x2
data_test['x3']=x3
data_test['x4']=[num2str(x) for x in x4]

y_1 = (3*x1)*(x2>15)+(-3)*x1*(x2<=15)+(-3*x2)*(x2>=10)-5*x2*(x2<10)+(x3)*(x1>10)\
    -x3*(x1<10)
y_2 = x3.copy()
y_2[numpy.where(x4==3)] = -3*x3[numpy.where(x4==3)]
y = y_1+y_2+numpy.random.randn(500)

data_test['y']=y
# variables setting
var_all=numpy.array(['x1', 'x2', 'x3', 'x4', 'y'])
var_res=['y']
var_split=['x1', 'x2', 'x3', 'x4']
var_reg=['x1', 'x2', 'x3']
id_res = numpy.where(numpy.isin(var_all,var_res))[0]
id_cand_all = numpy.where(numpy.isin(var_all,var_split))[0]
id_value_all = numpy.where(numpy.isin(var_all,var_reg))[0]
id_dis = numpy.where(numpy.isin(var_all,['x4']))[0]
'=============================2. simulated example============================='  
# 2.1 SLRT (Segmented Linear Regression Trees)           
# 2.1.1 with simple stopping rules
tree_slrt = lrt_simple.grow_tree(data_train, id_value_all, id_cand_all, id_dis, id_res, index = [1],\
                          labels = var_all, max_depth = 10, Nmin = 50, start = True, sNsplit = 500)
pred_slrt = tree_slrt.predict_all(data_test[['x1','x2','x3','x4']])
rmse(pred_slrt,data_test['y'])
# 2.1.2 with hypothesis based stopping
# fit model and predict
tree_lrt_wise_stop = lrt_testing.grow_tree(data_train, id_value_all, id_cand_all, id_dis, id_res, index = [1],\
                          labels = var_all, Nmin = 50, start = True, sNsplit = 500)
pred_slrt_wise_stop = tree_slrt.predict_all(data_test[['x1','x2','x3','x4']])
rmse(pred_slrt_wise_stop,data_test['y'])
# plot the tree
filename = '/Users/zhengxiangyu/Documents/Reasearch/RecursivePartition/5.Code_Examples/EXAMPLE_python_3.7/eg1/res_independent/0214SLRT_testing_stop.jpg'
writer1 = tw.treeWriter(tree_lrt_wise_stop)
writer1.write(filename)
Image(filename = filename, width=100, height=60)
# 2.2 CART (piece-wise linear regression tree)
data_train_onehot = pd.get_dummies(data_train[['x1','x2','x3','x4']], drop_first=True)
tree_cart = tree.DecisionTreeRegressor()
tree.plot_tree(tree_cart)
data_test_onehot = pd.get_dummies(data_test[['x1','x2','x3','x4']], drop_first=True)
pred_cart=tree_cart.predict(data_test_onehot)
rmse(pred_cart, data_test['y'])
