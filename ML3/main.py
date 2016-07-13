# coding:utf-8
from typing import TypeVar

import ML3.treePlotter

T = TypeVar('T')

__author__ = 'di_shen_sh@163.com'


fr = open('lenses.txt')
lenses = [tuple(inst.strip().split('\t')) for inst in fr.readlines()]
lensesLabels = ('age', 'prescript', 'astigmatic', 'tearRate')
lensesTree = ML3.calculate.create_decision_tree(lenses, lensesLabels)
# lensesTree = ML3.trees.create_tree(lenses, lensesLabels)
ML3.treePlotter.create_plot(lensesTree)