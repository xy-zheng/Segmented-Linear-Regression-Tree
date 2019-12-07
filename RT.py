#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 20:24:48 2018

@author: zhengxiangyu
"""

import scipy
import scipy.optimize
import scipy.stats
import numpy
import copy
import random
from bisect import bisect_right


class Tree(object):
    def __init__(self, error, predict, stdev, start, num_points, var_dis, var_res, index, region_info):
        self.error = error
        self.predict = predict
        self.stdev = stdev
        self.start =  start
        self.num_points = num_points
        self.index = index[0]
        self.var_dis = var_dis
        self.var_res = var_res
        'for plot ----------------------------'
        self.region_info = region_info
        'for plot ----------------------------'
        self.split_var = None
        self.split_val = None
        self.split_lab = None
        'for plot ----------------------------'
        self.split_info = None
        'for plot ----------------------------'
        self.left = None
        self.right = None

    def lookup_sig2(self, x):
        """Returns the standard error"""
        if self.left == None:
            # Attention! include all continuous vars in regression
            return (self.stdev)
        if isinstance(self.split_val,float) or isinstance(self.split_val,int):
            if x.loc[self.split_var] <= self.split_val:
                return self.left.lookup_sig2(x)
            else:
                return self.right.lookup_sig2(x)
        else:
            if x.loc[self.split_var] in self.split_val[0]:
                return self.left.lookup_sig2(x)
            else:
                return self.right.lookup_sig2(x)


    def lookup(self, x):
        """Returns the predicting beta parameter.
           here we require that x is a tuple"""
        if self.left == None:
            # Attention! include all continuous vars in regression
            return self.predict
        if isinstance(self.split_val,float) or isinstance(self.split_val,int):
            if x.loc[self.split_var] <= self.split_val:
                return self.left.lookup(x)
            else:
                return self.right.lookup(x)
        else:
            if x.loc[self.split_var] in self.split_val[0]:
                return self.left.lookup(x)
            else:
                return self.right.lookup(x)



    def predict_all(self, data):
        """Returns the predicted values for a dataframe, and returns a series."""
        return data.apply(lambda x:self.lookup(x), axis = 1)


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
            # numerically approximation error
            if alpha < 0:
                alpha = 0
            'for debug ----------------------------'
            #for node in nodes:
            #    print 'the node to be pruned is:', node.index
            'for debug ----------------------------'
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
        """Returns the length of the tree, which means the longest path."""
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

# !!! before invoking grow_tree, an global variable: kendall_pvalue_total should be defined
kendall_pvalue_total = {}
def grow_tree(data, id_cand_all, id_dis, id_res, index, labels = [1], depth = 1, \
              max_depth = 500, Nmin = 50,  start = False, feat_bag = False,\
              Nsplit = 100, region_info = 'All Data',mi=0.5):
    """Function to grow a regression tree given some training data."""
    """parameters of Tree: error, predict, stdev, start, num_points, index. """
    """Nsplit is to be passed in get_split_val; Nsplit is to be passed in get_split_var, but inside each are both called Nsplit"""
    """id_** means the position of ** names in labels """
    'for debug ----------------------------'
    #print index
    #print 'number of points is,', len(data)
    'for debug ----------------------------'
    # 0: if feature bagging (for random forests) choose sqrt(p) variables, where p is the total number of variables.
    #    otherwise select all variables.
    num_splitvars = len(id_cand_all)
    if (feat_bag):
        index_bag_cand = random.sample(range(num_splitvars), int(num_splitvars**(mi)))
    else:
        index_bag_cand = range(num_splitvars)
    id_cand = list(numpy.array(id_cand_all)[index_bag_cand])
    if len(id_dis)>0:
        var_dis = labels[id_dis]
    else:
        var_dis = numpy.array([])
    var_res = labels[id_res]
    pred_value = numpy.mean(numpy.array(data[var_res]))
    #print 'predict is', pred_value
    # get the information of linear regression on region: data
    root = Tree(region_error(data[var_res]), pred_value, numpy.std(numpy.array(data[var_res])), start, len(data), var_dis, var_res, index, region_info)

    # regions has fewer than Nmin data points
    if (len(data) <= 2*Nmin):
        'for debug ----------------------------'
        #print 'This is the leaf end and the standard error is', std
        'for debug ----------------------------'
        return root
    # length of tree exceeds max_depth
    if depth >= max_depth:
        'for debug ----------------------------'
        #print 'This is the leaf end and the standard error is', std
        'for debug ----------------------------'
        return root
    # 1: select the split var
    split_var, split_val = get_split(data, id_cand, id_dis, id_res, labels, Nmin = Nmin, Nsplit = Nsplit)
    'debug 0911---------------------------'
    #print 'split_var is', split_var
    'debug 0911---------------------------'
    if split_val == None:
        return root
    # 2: pass the information to root node
    root.split_var = split_var
    root.split_val = split_val
    root.split_lab = split_var
    root.index = index[0]
    'for plot ----------------------------'
    if split_var not in var_dis:
        root.split_info = root.split_lab + ': ' + str(float('%.2f' % split_val))
    else:
        root.split_info = root.split_lab + ': ' + str(split_val)
    'for plot ----------------------------'
    # 4: split dataset according to split_var&split_val
    if split_var in var_dis:
        id1 = data[split_var].isin(split_val [0])
        data1 = data.loc[id1].copy()
        id2= data[split_var].isin(split_val [1])
        data2 = data.loc[id2].copy()
    else:
        id1 = data[split_var] <= split_val
        data1 = data.loc[id1].copy()
        id2 = data[split_var] > split_val
        data2 = data.loc[id2].copy()
    'for debug ----------------------------'
    #print 'data1 has ', len(data1.values()), 'rows'
    #print 'data2 has ', len(data2.values()), 'rows'
    'for debug ----------------------------'
    # 5: grow right and left branches
    # index plus one after each growing
    index[0] = index[0]+1
    'for plot ----------------------------'
    if split_var not in var_dis:
        region_info_left = root.split_lab + '<' + str(round(split_val,2))
    else:
        set_str = set2str(split_val[0])
        region_info_left = root.split_lab + ' in ' + set_str
    'for plot ----------------------------'
    root.left = grow_tree(data1, id_cand_all, id_dis, id_res, index, labels = labels,\
                          depth = depth + 1, max_depth = max_depth, Nmin = Nmin, \
                          feat_bag = feat_bag, region_info = region_info_left,mi=mi)
    index[0] = index[0]+1
    'for plot ----------------------------'
    if split_var not in var_dis:
        region_info_right = root.split_lab + '>' + str(round(split_val,2))
    else:
        set_str = set2str(split_val[1])
        region_info_right = root.split_lab + ' in ' + set_str
    'for plot ----------------------------'
    root.right = grow_tree(data2, id_cand_all, id_dis, id_res, index, labels = labels,\
                          depth = depth + 1, max_depth = max_depth, Nmin = Nmin, \
                          feat_bag = feat_bag, region_info = region_info_right,mi=mi)
    return root

def cvt(data, id_cand_all, id_dis, id_res, v, labels = [], max_depth = 500, Nmin = 50, Nsplit = 10, mi=0.5):
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
    full_tree = grow_tree(data, id_cand_all, id_dis, id_res, [1], labels = labels, \
                          start = True, Nmin = Nmin, Nsplit = Nsplit,mi=mi)
    full_a, full_t = full_tree.prune_tree()
    var_res = labels[id_res].tolist()
    # ak' = (ak * ak+1)^(1/2)
    a_s = []
    for i in range(len(full_a) - 1):
        a_s.append((full_a[i] * full_a[i+1])**(.5))
    a_s.append(full_a[-1])
    #print a_s
    # stratify data, sort data by the rank of y
    # break the data into v subsamples of roughly equal size,
    # [i::v] means start from i and take a step of v to get items
    # lv_s is a list of length v, each element is a dictionary (a subsumple)
    lv_s = [range(i, len(data), v) for i in range(v)]
    # list of tree sequences for each training set
    t_vs = []
    # list of testing data for each training set
    test_vs = []
    # list of alpha values for each training set
    alpha_vs = []

    # grow and prune each training set
    for i in range(len(lv_s)):
        if i==0:
            train_id = reduce(lambda x,y:x+y,lv_s[i+1:])
        elif i==(len(lv_s)-1):
            train_id = reduce(lambda x,y:x+y,lv_s[0:i])
        else:
            train_id = reduce(lambda x,y:x+y,lv_s[0:i])+reduce(lambda x,y:x+y,lv_s[i+1:])
        test_id = lv_s[i]
        train = data.iloc[train_id, ]
        test = data.iloc[test_id, ]
        full_tree_v = grow_tree(train, id_cand_all, id_dis, id_res, [1], labels = labels, \
                                max_depth = max_depth, Nmin = Nmin, start = True,mi=mi)
        alphas_v, trees_v = full_tree_v.prune_tree()
        '------------------check-------------------------'
        t_vs.append(trees_v)
        alpha_vs.append(alphas_v)
        test_vs.append(test)


    # choose the subtree that has the minimum cross-validated
    # error estimate
    min_R = float("Inf")
    min_ind = 0
    # the length of pruned full_tree
    for i in range(len(full_t)):
        ak = a_s[i]
        R_k = 0
        # number of cross validation folds
        for j in range(len(t_vs)):
            # closest alpha value in sequence v to
            # alphak
            a_vs = alpha_vs[j]
            tr_vs = t_vs[j]
            # find the largest one that <= ak in a_vs
            alph_ind = bisect_right(a_vs, ak) - 1
            test_data = test_vs[j]
            pred_vals = tr_vs[alph_ind].predict_all(test_data)
            true_vals = (test_data[var_res]).iloc[:,0]
            r_kv = numpy.std(numpy.array(true_vals) - numpy.array(pred_vals))
            #print r_kv
            R_k = R_k + r_kv
        if (R_k < min_R):
            min_R = R_k
            min_ind = i
    return full_t[min_ind]




"""below are the two main functions for split_var and split_val"""
def get_split(data, id_cand, id_dis, id_res, labels, Nmin ,Nsplit):
    # data is a dataframe iterate over all variables
    # 1. get kendall taus for all candidate split variables
    # 1.0 get data prepared
    # 1.1 get candidate split values
    var_split = labels[id_cand]
    if len(id_dis)>0:
        var_dis = labels[id_dis]
    else:
        var_dis = numpy.array([])
    var_res = labels[id_res]
    split_dict = get_split_dict(var_split, var_dis, data, Nsplit, Nmin)
    # 1.2 fill in kendall_pvalue_total
    min_error = -1
    for var in split_dict:
        for val in split_dict[var]:
            if var in var_dis:
                error = error_function_dis(split_val = val, split_var = var, var_res = var_res, data = data, Nmin = Nmin)
            else:
                error = error_function_con(split_val = val, split_var = var, var_res = var_res, data = data, Nmin = Nmin)
            if((error < min_error and error > 0) or (min_error == -1)):
                min_error = error
                split_val = val
                split_var = var
    return  split_var, split_val



"""below are functions used to calculate error and kendall pvalue,
   will be called for figuring out the best split_var and split_val"""

def region_error(res_data):
    """Calculates sum of squared error for some node in the regression tree."""
    res = numpy.array(res_data)
    return numpy.sum((res - numpy.mean(res))**2)


def error_function_con(split_val, split_var, var_res, data, Nmin):
    """Function calculate the error of one node, given split_point/value.
       split_* is int, data is dictionary."""
    id1 = data[split_var] <= split_val
    data1 = data.loc[id1].copy()
    id2 = data[split_var] > split_val
    data2 = data.loc[id2].copy()
    if len(data1) > Nmin and len(data2) > Nmin:
        return -1
    else:
        return region_error(data1[var_res]) + region_error(data2[var_res])


def error_function_dis(split_val, split_var, var_res, data, Nmin):
    """Function calculate the error of one node, given split_val/value.
       split_var is discrete variable, data is dictionary."""
    id1 = data[split_var].isin(split_val [0])
    data1 = data.loc[id1].copy()
    id2 = data[split_var].isin(split_val [1])
    data2 = data.loc[id2].copy()
    if len(data1) > Nmin and len(data2) > Nmin:
        return -1
    else:
        return region_error(data1[var_res]) + region_error(data2[var_res])

"""var_split and var_dis are both numpy.array"""
def get_split_dict(var_split, var_dis, data, Nsplit, Nmin):
    """function to return a dictionary of candidate splits
        where the keys are split variables and the corres. values are split points"""
    split_dict = {}
    for var in var_split:
        if var not in var_dis:
            x_split_list = data[var].values
            """for ordered categorical variable"""
            if(len(set(x_split_list))<Nsplit):
                split_range=numpy.array(list(set(x_split_list)))
            else:
                x_split_ranked = numpy.sort(x_split_list)
                id_str = Nmin
                id_end = len(x_split_list)-Nmin
                step = (len(x_split_list) - 2*Nmin)/Nsplit
                if step == 0:
                    step = 1
                id_split = numpy.arange(id_str, id_end, step)
                split_range = x_split_ranked[id_split]
            split_dict.update({var:split_range})
        else:
            x_split_set = list(set(data[var].values))
            split_range = list(paired_power_set(x_split_set))
            split_dict.update({var:split_range})
    return split_dict

def paired_power_set(s):
    """This function returns an iterator, each value is a pair of sets"""
    """they are complementary, either is complete or empty"""
    n = len(s)
    test_marks = [1<<i for i in range(0,n)] # <<i 相当于 *2^{i}
    pair_1 = range(1, 2**n-1)[0::2]
    pair_2 = range(1, 2**n-1)[1::2]
    pair_2.reverse()
    pair = zip(pair_1, pair_2)
    for k1, k2 in pair:
        l1 = []
        l2 = []
        for idx, item in enumerate(test_marks): # enumerate returns a iterator
            if k1&item:
                l1.append(s[idx])
            if k2&item:
                l2.append(s[idx])
        yield [set(l1),set(l2)]
def set2str(set):
    if len(set)==1:
        return '{ '+str(list(set)[0])+' }'
    else:
        return '{ '+reduce(lambda x,y: str(x)+', '+str(y), set)+' }'

"""below are functions to get information from growned trees"""
def get_path(tree, li, path):
    """Get a list of all paths, where each element is one single path
    corresponding to a node.
    when call the function, the parameters shoulde be [] and [], for eg,
    res = get_paths(tree,[],[])."""

    if tree.right != None:
        if isinstance(tree.split_val, float):
            li.append(tree.split_lab+'<'+str(tree.split_val))
            path=get_path(tree.left, li, path)
            li.pop()
            li.append(tree.split_lab+'>'+str(tree.split_val))
            path=get_path(tree.right, li, path)
            li.pop()
            return path
        else:
            li.append(tree.split_lab+' in '+str(tree.split_val[0]))
            path=get_path(tree.left, li, path)
            li.pop()
            li.append(tree.split_lab+' in '+str(tree.split_val[1]))
            path=get_path(tree.right, li, path)
            li.pop()
            return path
    else :
        path.append(copy.deepcopy(li))
        return path

# get the index set of all terminal/leaf nodes
def get_leaf_index(tree, leaves):
    """Get the index of all leaves. when function is called, leaves should be
        set as [],see: get_leaf_index(tree, [])"""
    if tree.right != None:
        leaves = get_leaf_index(tree.left, leaves)
        leaves = get_leaf_index(tree.right, leaves)
        return leaves
    else:
        leaves.append(tree.index)
        return leaves

# get a dictionary of predictive parameters beta all terminal/leaf nodes
def get_leaf_beta(tree, betas):
    """key: leaf index ; value: beta
        the order is the same as leaves"""
    if tree.right != None:
        betas = get_leaf_beta(tree.left, betas)
        betas = get_leaf_beta(tree.right, betas)
        return betas
    else:
        betas.update({tree.index: tree.predict})
        return betas

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
        if isinstance(tree.split_val, float):
            if x[tree.split_var] < tree.split_val:
                return get_matching_node(tree.left, x)
            else:
                return get_matching_node(tree.right, x)
        else:
            if x[tree.split_var] in tree.split_val[0]:
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
