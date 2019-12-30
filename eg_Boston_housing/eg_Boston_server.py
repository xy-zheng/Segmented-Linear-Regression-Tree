#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: zhengxiangyu
"""
import os
import sys
import random
import numpy
import pandas as pd
import time
import math
sys.path.append('/home/xiangyu/tree_methods/3.share/')
import SLRT_faster as lrt
import RT as rt
import SLRT_RF as lrf
import RT_RF as rf
import tree_lasso as tlasso
'----------------------1. data process---------------------------------'
'======================1.1 read data-----------------------------------'
data0=pd.read_csv('Boston.csv')
data1=data0.dropna()
data1.loc[:,'log_medv']=data1['medv'].apply(math.log)
var_all=numpy.array(['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
       'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat', 'log_medv'])
var_res=['log_medv']
var_split=['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
       'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
var_reg=['crim', 'zn', 'rad','indus', 'nox', 'rm', 'age',
       'dis',  'tax', 'ptratio', 'black', 'lstat']
id_res = numpy.where(numpy.isin(var_all,var_res))[0]
id_cand_all = numpy.where(numpy.isin(var_all,var_split))[0]
id_value_all = numpy.where(numpy.isin(var_all,var_reg))[0]
id_dis = numpy.where(numpy.isin(var_all,['chas']))[0]
'======================1.2 prepare crossvalidation data----------------'
data1.sort_values('log_medv',inplace = True)
lv_s = [range(i, len(data1), 10) for i in range(10)]
test_sets=[]
train_sets=[]
for i in range(len(lv_s)):
    if i==0:
        train_id = reduce(lambda x,y:x+y,lv_s[i+1:])
    elif i==(len(lv_s)-1):
        train_id = reduce(lambda x,y:x+y,lv_s[0:i])
    else:
        train_id = reduce(lambda x,y:x+y,lv_s[0:i])+reduce(lambda x,y:x+y,lv_s[i+1:])
    test_id = lv_s[i]
    train = data1.iloc[train_id, ]
    test = data1.iloc[test_id, ]
    train_sets.append(train)
    test_sets.append(test)
'--------------------1. check the mspe of models via cross-validation-------------------------------------'
Nmin_cv=60
'====================1.1 piecewise linear tree model + lasso========================'
random.seed(1234)
mspe_cv1=[]
mspe_cv2=[]
alpha_min_cv=-3
alpha_max_cv=-1
start=time.clock()
for i in range(10):
    data_train = train_sets[i]
    data_test = test_sets[i]
    random.seed(1234)
    model_lrt = lrt.grow_tree(data_train, id_value_all, id_cand_all, id_dis, id_res, index = [1],\
                          labels = var_all, Nmin = Nmin_cv, start = True, Nsplit = 100, \
                           sNsplit = 10)
    mspe1=numpy.mean((model_lrt.predict_all(data_test)-data_test['log_medv'])**2)
    random.seed(1234)
    mspe2=tlasso.tree_lasso(model_lrt,data_train,data_test,var_reg,var_res,
                          alpha_min=alpha_min_cv, alpha_max=alpha_max_cv)
    mspe_cv1.append(mspe1)
    mspe_cv2.append(mspe2)
end=time.clock()
print end-start
rmspe_cv1=math.sqrt((numpy.sum(mspe_cv1)-max(mspe_cv1)-min(mspe_cv1))/(len(mspe_cv1)-2))
rmspe_cv2=math.sqrt((numpy.sum(mspe_cv2)-max(mspe_cv2)-min(mspe_cv2))/(len(mspe_cv2)-2))
print rmspe_cv1
print rmspe_cv2

file1=open('boston_results.txt','w')
file1.write('lrt:'+str(rmspe_cv1)+'\n')
file1.write('lrt_lasso'+str(rmspe_cv2)+'\n')
file1.close()
'====================1.2 piecewise-constant tree model========================'
mspe_cv3=[]
for i in range(10):
    data_train = train_sets[i]
    data_test = test_sets[i]
    model_rt = rt.grow_tree(data_train, id_cand_all, id_dis, id_res, index = [1],\
                          labels = var_all, Nmin = Nmin_cv, start = True, Nsplit = 100)
    mspe3=numpy.mean((model_rt.predict_all(data_test)-data_test['log_medv'])**2)
    mspe_cv3.append(mspe3)

rmspe_cv3=math.sqrt((numpy.sum(mspe_cv3)-max(mspe_cv3)-min(mspe_cv3))/(len(mspe_cv3)-2))
print rmspe_cv3

file1=open('boston_results.txt','a')
file1.write('cart'+str(rmspe_cv3)+'\n')
file1.close()
