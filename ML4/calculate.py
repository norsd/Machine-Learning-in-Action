# coding:utf-8

from numpy import *
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar
# from norlib.Dictionary import *

__auth__ = 'di_shen_sh@gmail.com'

T = TypeVar('T')

def calculate_native_bayes(a_data_set, a_labels):
    """
    计算每个单词属于特定label的概率
    @param a_data_set:
    datas = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
        ]
    @param a_labels:
    labels = [0, 1, 0, 1, 0, 1]
    @return:
    {
    0:{ dog: 0.8, mr: 0.2, ...}, 某个词在label0中的概率: P(wi|label0)
    1:{},
    }
    """
    word_set = set()
    for words in a_data_set:
        word_set |= set(words)

    template_word_2_count = {}
    for word in word_set:
        template_word_2_count[word] = 0

    dt_dt_label = {}
    label_2_word_2_count = {}
    label_2_sample_count = {}
    for label in a_labels:
        dt_dt_label.setdefault(label, {})
        label_2_word_2_count.setdefault(label, template_word_2_count.copy())
        label_2_sample_count.setdefault(label, 0)

    for i in range(0, len(a_data_set)):
        label = a_labels[i]
        words = a_data_set[i]
        label_2_sample_count[label] += len(words)
        for w in words:
            label_2_word_2_count[label][w] += 1

    for label in a_labels:
        dt_dt_label.setdefault(label, {})
        dt_label = dt_dt_label[label]
        sample_count = label_2_sample_count[label]
        for word in word_set:
            word_count = label_2_word_2_count[label][word]
            dt_label[word] = word_count/float(sample_count)
    return dt_dt_label

    # dt_all = {}
    # for i in range(0, len(a_data_set)):
    #     words = a_data_set[i]
    #     label = a_labels[i]
    #     dt_label = dt_dt_label[label]
    #     for w in words:
    #         if w in dt_all:
    #             dt_all[w] += 1
    #         else:
    #             dt_all[w] = 1
    #         if w in dt_label:
    #             dt_label[w] += 1
    #         else:
    #             dt_label[w] = 1
    # # 0:{dog: 0.8, has: 0.2, ...}, 每个词属于label0的概率
    # # 1:{}
    # label_word_p = {}
    # for label in a_labels:
    #     if label not in label_word_p:
    #         label_word_p[label] = {}
    #         for word in dt_all.keys():
    #             if word not in dt_dt_label[label]:
    #                 label_word_p[label][word] = 0
    #             else:
    #                 label_word_p[label][word] = dt_dt_label[label][word]/float(dt_all[word])
    # return label_word_p


    # dt_ret = {}
    # for i in range(0, len(a_labels)):
    #     dt_ret.setdefault(a_labels[i], {})
    #
    # # 每个词汇出现的个数
    # # 例如:
    # # { dog:{ label0: 2, label1: 5, label2: 5},
    # #   cat:{ label0: 1, label1: 9, label2: 0},
    # #   ......
    # # }
    # dt_count = {}
    # dt_label_count = {}
    # set_word = set([])
    # set_label = set(a_labels)
    # for i in range(0, len(a_data_set)):
    #     label = a_labels[i]
    #     row_data = a_data_set[i]
    #     set_word.update(row_data)
    # for word in set_word:
    #     dt_count[word] = {}
    #     for label in set_label:
    #         dt_count[word][label] = 0
    # for i in range(0, len(a_data_set)):
    #     label = a_labels[i]
    #     row_data = a_data_set[i]
    #     for data in row_data:
    #         dt_count[data]["all"] += 1
    #         dt_count[data][label] += 1
    #
    #     dt_count.setdefault(label, 0)
    #     dt_count[label] += len(data)
    #     for w in data:
    #         dt = get_value(dt_ret, label, {})
    #         dt.setdefault(w, 0)
    #         dt[w] += 1
    # for (k, v) in dt_root.items():
    #     dt = v
    #     label = k
    #     for (kw, vw) in dt.items():
    #         word = kw
    #         dt[word] /= float(dt_count[label])
    # return tuple(dt_root.values())



class NativeByes:
    dt_dt_label = {}

    def __init__(self):

        return

    def add_samples(self, a_data_set, a_labels):
        """
        计算每个单词属于特定label的概率
        @param a_data_set:
        datas = [
           ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
           ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
           ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
           ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
           ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
           ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
           ]
        @param a_labels:
        labels = [0, 1, 0, 1, 0, 1]
        @return:
        {
        0:{ dog: 0.8, mr: 0.2, ...}, 某个词在label0中的概率: P(wi|label0)
        1:{},
        }
        """
        word_set = set()
        for words in a_data_set:
            word_set |= set(words)

        template_word_2_count = {}
        for word in word_set:
            template_word_2_count[word] = 0

        dt_dt_label = {}
        label_2_word_2_count = {}
        label_2_sample_count = {}
        for label in a_labels:
            dt_dt_label.setdefault(label, {})
            label_2_word_2_count.setdefault(label, template_word_2_count.copy())
            label_2_sample_count.setdefault(label, 0)

        for i in range(0, len(a_data_set)):
            label = a_labels[i]
            words = a_data_set[i]
            label_2_sample_count[label] += len(words)
            for w in words:
                label_2_word_2_count[label][w] += 1

        for label in a_labels:
            dt_dt_label.setdefault(label, {})
            dt_label = dt_dt_label[label]
            sample_count = label_2_sample_count[label]
            for word in word_set:
                word_count = label_2_word_2_count[label][word]
                dt_label[word] = word_count / float(sample_count)
        return







