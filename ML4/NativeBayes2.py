# coding:utf-8
from numpy import *
from norlib.Exception import *
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar



__auth__ = 'di_shen_sh@gmail.com'

T = TypeVar('T')


class NativeBayes2:
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
        self.vocable_set = set()
        word_count = 0
        for i in range(len(a_data_set)):
            self.vocable_set |= set(a_data_set[i])
        self.vocable_list = list(self.vocable_set)
        self.vocable_list.sort()

        self.vocable_index2name = {}
        self.vocable_name2index = {}
        for i in range(0, len(self.vocable_list)):
            self.vocable_index2name.setdefault(i, self.vocable_list[i])
            self.vocable_name2index.setdefault(self.vocable_list[i], i)

        self.pr_label = {}
        label2word_count = {}  # 记录每一个label拥有word的个数
        for label in a_labels:
            self.pr_label[label] = 0
        for label in self.pr_label:  # 注意是dictionary,所以label不会重复
            label2word_count[label] = len(self.vocable_list)

        self.vocable_pr_array = ones(len(self.vocable_list)) + ones(len(self.vocable_list))# 先填入每一个词汇出现的个数
        for i in range(len(a_data_set)):
            label = a_labels[i]
            datas = self.__words_to_standard_sample_vector(a_data_set[i])
            label2word_count[label] += sum(datas)
            vector = self.label2vocable_pr_array.setdefault(label, ones(len(self.vocable_list)))
            vector += datas  # 属于特定label的word个数记录
            self.vocable_pr_array += datas  # 全局的word个数记录
        total_word_count = sum(self.vocable_pr_array)  # 当前vocable_pr_array中记录的还是全局的word个数
        self.vocable_pr_array /= float(total_word_count)  # 得到每一个词汇的全局概率
        # self.vocable_pr_array /= len(a_data_set)  # 得到每一个词汇的全局概率

        for label in self.label2vocable_pr_array:
            self.label2vocable_pr_array[label] /= float(label2word_count[label])
            self.pr_label[label] = label2word_count[label] / float(total_word_count)

    def __filter_words(self, a_words: List[str])->List[str]:
        """
        从a_words中过滤掉vocable_set不存在的
        :param a_words:
        :return: 过滤掉之后的List
        """
        return [w for w in a_words if w in self.vocable_set]

    def __words_to_standard_sample_vector(self, a_words: List) -> array:
        """
        帮助函数
        将一个样本a_words转化为标准样本序列
        a_words中出现的word，将在标准样本序列中置为n(n>0),其他word将为0
        如果a_words中的某一个word不存在于样本中,将抛出异常
        :param a_words:
        :return: 返回标准样本序列,类型为Numpy.array
        """
        vector = zeros(len(self.vocable_set))
        for word in a_words:
            if word in self.vocable_set:
                vector[self.vocable_name2index[word]] += 1
            else:
                raise "样本中没有{0}这个词".format(word)
        return vector

    def __to_word_pr_in_label(self, a_words: List, a_label: object) -> array:
        """
        返回a_words中每一个词在样本中且类别为a_label的概率
        如果某一个词不在样本中则抛出异常
        :param a_words: 需要得到概率的词的集合
        :param a_label: 样本中的类别
        :return: a_words中的每个词在样本中且类别为a_label的概率,类型为Numpy.array
        """
        vector = zeros(len(a_words))
        for i in range(0, len(a_words)):
            word = a_words[i]
            if word in self.vocable_set:
                pr = self.pr_word_in_label[a_label][word]
                vector[i] = pr
            else:
                raise StringException("样本中没有{0}这个词", word)
        return vector

    def __to_word_pr(self, a_words: List) -> array:
        """
        返回a_words中每一个词在样本中的概率
        如果某一个词不在样本中则抛出异常
        :param a_words: 需要得到概率的词的集合
        :return:  a_words中的每个词在样本中的概率,类型为Numpy.array
        """
        vector = zeros(len(a_words))
        for i in range(0, len(a_words)):
            word = a_words[i]
            if word in self.vocable_set:
                index = self.vocable_name2index[word]
                pr = self.vocable_pr_array[index]
                vector[i] = pr
            else:
                raise StringException("样本中没有{0}这个词", word)
        return vector

    def test(self, a_ω) -> Dict:
        """
        计算向量a_ω属于各个label的概率
        :param a_ω:
        :return: a_ω属于各个label的概率
        """
        ω = self.__filter_words(a_ω)
        dt_ret = {}
        error_code_overflow = False  # 设为True将观测到下溢出
        #  由于很多小数相乘引起的下溢出
        if error_code_overflow:
            pr_w = prod(self.__to_word_pr(ω))
            print("pr_w: {0}".format(pr_w))
            for l in self.pr_label:
                word_vector = self.__to_word_pr_in_label(ω, l)
                pr = prod(word_vector) * self.pr_label[l]
                print("{0} pr_word:{1}".format(l, prod(word_vector)))
                print("{0} pr_word*pr_label:{1}".format(l, pr))
                print("pr > pr_w ?: {0}, delta:{1} (理论上应该总是为False)".format(pr > pr_w, pr-pr_w))
                pr = pr / pr_w
                dt_ret[l] = pr
        else:
            pr_w_arr = self.__to_word_pr(ω)
            p = 0
            label = None
            for l in self.pr_label:
                word_vector = self.__to_word_pr_in_label(ω, l)
                pr_arr = word_vector * self.pr_label[l]
                pr_arr = pr_arr / pr_w_arr
                dt_ret[l] = prod(pr_arr)
                if dt_ret[l] > p:
                    label = l
                    p = dt_ret[l]
                # print("{0}: w|l:      {1}".format(l, word_vector))
                # print("{0}: label_pr: {1}".format(l, self.pr_label[l]))
                # print("{0}: pr_arr:   {1}".format(l, pr_arr))
                # print("{0}: pr_w_arr: {1}".format(l, pr_w_arr))
                # print("{0}: result:   {1}".format(l, test_arr))
        return (dt_ret, label)






