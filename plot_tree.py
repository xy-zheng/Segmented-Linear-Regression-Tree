#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 16:44:08 2018

@author: zhengxiangyu
"""
import pygraphviz as pgv
""" A is pygraphviz.agraph.AGraph type"""
class treeWriter():
    def __init__(self, tree):
        self.tree = tree
        self.A = pgv.AGraph(directed=True,strict=True)
    def write(self, outfile):
        def  writeHelp(root, A):
            # when a node is not a leaf, it has split_info
            node0 = root.index
            if root.left == None:
                A.add_node(node0, label = root.region_info, style = 'filled', fontsize=50, height=1.5,fillcolor = 'palegreen', color = 'palegreen')
                '---------add std error-----------------'
                #leaf_error_node = str(root.index)+'_sigma'
                #A.add_node(leaf_error_node, label = 'sigma='+str(round(root.stdev,2)), color = 'white')
                #A.add_edge(node0, leaf_error_node, weight = 100, color = 'palegreen')
                '---------add std error-----------------'
            else:
                # midle root
                A.add_node(node0, label = root.region_info, shape = 'box', fontsize=50, height=1.5,style = 'filled', fillcolor = 'lemonchiffon')
                # left side
                lnodes = writeHelp(root.left, A)
                A.add_edge(node0, lnodes,minlen =3, penwidth=3, arrowsize=2)#+ str(round(root.split_val,2)))
                # right side
                rnodes = writeHelp(root.right, A)
                A.add_edge(node0, rnodes,minlen =3, penwidth=3, arrowsize=2)#+ str(round(root.split_val,2)))
            return node0
        writeHelp(self.tree, self.A)
        self.A.graph_attr['epsilon']='0.1'
        #print self.A.string() # print dot file to standard output
        self.A.layout('dot') # layout with dot
        self.A.draw(outfile) # write to file
#    def graphviz_string(self):
#        return self.A.string()
