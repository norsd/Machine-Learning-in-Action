# coding:utf-8

from typing import List
from typing import Tuple
from typing import TypeVar

import ML4.NativeBayes
import ML4.NativeBayes2

__author__ = 'di_shen_sh@gmail.com'


def create_data_set()->Tuple[List, List]:
    """
    返回2个数组
    1个含有6个子数组
    1个含有6个0或者1的长度为6的int[]
    数值0表示正常词汇
    数值1表示侮辱性词汇
    @return:
    :rtype: tuple(List[str], List[float])
    """
    datas = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
        ]
    labels = [0, 1, 0, 1, 0, 1]
    return datas, labels

#  nb = ML4.NativeBayes.NativeBayes()
nb = ML4.NativeBayes2.NativeBayes2()
(datas, label) = create_data_set()
nb.add_samples(datas, label)
print(nb.pr_label)
print(nb.pr_word_in_label[label[0]])
print(nb.pr_word_in_label[label[1]])
result = nb.test(['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'])

print(result)

