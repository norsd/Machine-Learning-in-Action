# coding:utf-8
from typing import TypeVar

import ML3.treePlotter

T = TypeVar('T')

__author__ = 'di_shen_sh@163.com'

data_set = [
    (1, 1, 'maybe'),
    (1, 1, 'yes'),
    (1, 1, 'yes'),
    (1, 0, 'no'),
    (0, 1, 'no'),
    (0, 1, 'no'),
]

property_names = ('no surfacing', 'flippers')
entropy = ML3.calculate.calculate_shannon_entropy(data_set)
print(entropy)

tuple_entropy = ML3.calculate.calculate_feather_sum_entropy(data_set)
print(tuple_entropy)


test_split = ML3.calculate.split_data_set(data_set, 0, 1)
print(test_split)

test_best_split = ML3.calculate.choose_best_feather_split(data_set)
print(test_best_split)

dt = ML3.calculate.create_decision_tree(data_set, tuple(property_names))
print(dt)