#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:33:10 2018

@author: zhengxiangyu
"""
import SLRT_faster as lrt
import numpy
from sklearn import linear_model
def tree_lasso(tree, data_train, data_test, var_reg, var_res, alpha_min=-3, alpha_max=-1):
    leaf_index=lrt.get_leaf_index(tree,[])
    data_part=lrt.data_partition(tree,data_train)
    data_part_test=lrt.data_partition(tree,data_test)
    model_lrt_lasso={}
    model_paras={}
    for leaf in leaf_index:
        data_temp=data_part[leaf]
        data_x=data_temp[var_reg].values
        data_y=data_temp[var_res].iloc[:,0].values
        alpha=numpy.logspace(alpha_min, alpha_max, 100)  #穷举40个alpha
        reg=linear_model.LassoCV(alphas=alpha)
        reg.fit(data_x,data_y)
        model_lrt_lasso.update({leaf:reg})
        paras=list(reg.coef_)
        paras.insert(0,reg.intercept_)
        model_paras.update({leaf:paras})
        sse=0
    for leaf in leaf_index:
        data_temp=data_part_test[leaf]
        if len(data_temp)>0:
            data_x=data_temp[var_reg].values
            data_y=data_temp[var_res].values
            pred_y=model_lrt_lasso[leaf].predict(data_x)
            data_part_test[leaf].loc[:,'pred']=pred_y
            sse=sse+sum((data_temp[var_res].iloc[:,0].values-pred_y)**2)
    mspe_lrt_lasso=sse/len(data_test)
    return mspe_lrt_lasso