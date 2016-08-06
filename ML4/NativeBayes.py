# coding:utf-8
from numpy import *
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar

__auth__ = 'di_shen_sh@gmail.com'

T = TypeVar('T')


class NativeBayes:
    #  注意描述的语句:
    #  "my", "my", "dog"
    #  上面有2个词汇(vocable)分别是"my","dog"
    #  上面有3个词(word)
    pr_word_in_label = {}  # 每一个词属于特定label的概率 P(word|label)
    label2vocable_pr_array = {}  # 每一个词属于特定label的概率 P(word|label),注意是[label, array(0.2, 0.015, 0.87)]这种形式,用于后续计算
    pr_label = {}  # 每一个分类的概率P
    vocable_pr_array = zeros(0)  # 每一个词汇的概率
    vocable_list = []   # 词汇列表
    vocable_set = set()  # 词汇哈希表
    vocable_name2index = {}  # 记录单词在vocable_list在某一个index代表的word,例如: [1] = "cat"
    vocable_index2name = {}

    def __init__(self):
        dt_dt_label = {}
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
        self.__create_vocable(a_data_set, a_labels)
        for label in self.label2vocable_pr_array:
            vector = self.label2vocable_pr_array[label]
            # word_count = sum(vector)
            # vector_pr = vector/word_count
            for i in range(0, len(vector)):
                self.pr_word_in_label.setdefault(label, {})[self.vocable_index2name[i]] = vector[i]
        return

    def __create_vocable(self, a_data_set, a_labels):
        self.pr_label = {}
        label2word_count = {}  # 记录每一个label拥有word的个数
        for label in a_labels:
            self.pr_label.setdefault(label, 0)
            self.pr_label[label] += 1
        for label in self.pr_label:
            self.pr_label[label] /= float(len(a_labels))
            label2word_count[label] = 0  # 初始化

        self.vocable_set = set()
        word_count = 0
        for i in range(len(a_data_set)):
            words = a_data_set[i]
            label = a_labels[i]
            label2word_count[label] += len(words)  # 得到label2word_count
            self.vocable_set |= set(words)
        self.vocable_list = list(self.vocable_set)
        self.vocable_list.sort()
        self.vocable_pr_array = zeros(len(self.vocable_list))

        self.vocable_index2name = {}
        self.vocable_name2index = {}
        for i in range(0, len(self.vocable_list)):
            self.vocable_index2name.setdefault(i, self.vocable_list[i])
            self.vocable_name2index.setdefault(self.vocable_list[i], i)

        self.vocable_pr_array = zeros(len(self.vocable_list))  # 先填入每一个词汇出现的个数
        for i in range(len(a_data_set)):
            label = a_labels[i]
            datas = self.__to_word_vector(a_data_set[i])
            vector = self.label2vocable_pr_array.setdefault(label, zeros(len(self.vocable_list)))
            vector += datas  # 属于特定label的word个数记录
            self.vocable_pr_array += datas  # 全局的word个数记录
        total_word_count = sum(self.vocable_pr_array)  # 当前vocable_pr_array中记录的还是全局的word个数
        # 下面的算法错误,计算每一个词汇的全局概率不应该想当然的把词c出现的概率除以总的词汇数
        # 注意一个样本是一个数组
        # 所以应该除以样本数，也就是数组的个数
        # 如果除以总的词汇数, 则与之前计算p(c|l1)时将矛盾
        # p(c|l1)中是用数组的个数,而不是属于l1的词汇数
        # self.vocable_pr_array /= float(total_word_count)  # 得到每一个词汇的全局概率
        self.vocable_pr_array /= len(a_data_set)  # 得到每一个词汇的全局概率

        for label in self.label2vocable_pr_array:
            self.label2vocable_pr_array[label] /= float(label2word_count[label])

    def __to_word_vector(self, a_words):
        vector = zeros(len(self.vocable_set))
        for word in a_words:
            if word in self.vocable_set:
                vector[self.vocable_name2index[word]] += 1
            else:
                raise "a_words中含有样本中没有的单词"
        return vector

    def __to_word_pr_in_label(self, a_words, a_label):
        vector = zeros(len(a_words))
        for i in range(0, len(a_words)):
            word = a_words[i]
            if word in self.vocable_set:
                pr = self.pr_word_in_label[a_label][word]
                vector[i] = pr
            else:
                raise "a_words中含有样本中没有的单词"
        return vector

    def __to_word_pr(self, a_words):
        vector = zeros(len(a_words))
        for i in range(0, len(a_words)):
            word = a_words[i]
            if word in self.vocable_set:
                index = self.vocable_name2index[word]
                pr = self.vocable_pr_array[index]
                vector[i] = pr
            else:
                raise "a_words中含有样本中没有的单词"
        return prod(vector)

    def test(self, a_ω):
        dt_ret = {}
        pr_w = self.__to_word_pr(a_ω)
        for l in self.pr_label:
            word_vector = self.__to_word_pr_in_label(a_ω, l)
            pr = prod(word_vector) * self.pr_label[l]
            pr = pr / pr_w
            dt_ret[l] = pr
        return dt_ret






