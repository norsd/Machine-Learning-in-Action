# coding:utf-8

from numpy import *
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar

__auth__ = 'di_shen_sh@gmail.com'

T = TypeVar('T')


def calculate_shannon_entropy(a_data_set: List[Tuple[int, int, str]]):
    """
    计算数据集的熵
    @param a_data_set:
        List[ Tuple[ int, int, ..., str] ]
        每一个Tuple[ int, int, ..., str]代表一个样本,
        最后一个Column表示这个样本是哪一个对象
        其他每一个Column表示一个属性Property
        例如
        最后一个Column可能是"老虎，猫，鸟"
        其他的Column属性分别为: 是否汪汪叫， 几只脚， 是否有胡须
        计算 -∑p(i)logp(i)，
        p(i)是分类i的概率
        p(i) = 样本中分类为i的个数 / 样本总数
        不确定函数f = -logp(i)
        概率越大的事件，信息熵越小
        https://zh.wikipedia.org/wiki/%E7%86%B5_(%E4%BF%A1%E6%81%AF%E8%AE%BA)
        H = -∑E[logp(i)]
        例如:
        抛掷一个均匀的硬币10次，5次正面,5次反面
        则 H = -0.5xlog(0.5) - 0.5xlog(0.5) = -log(1/2) = -(log1 - log2) = -(0 - 1) = 1
        表示这个事件的信息量是1个bit
    @return:
    """
    count = len(a_data_set)
    dt_kind = {}
    for item in a_data_set:
        str_category = item[-1]  # 此样本所属分类,例如:"老虎,猫,鸟"
        dt_kind[str_category] = dt_kind.get(str_category, 0) + 1
    sum_entropy = 0.0
    for str_category in dt_kind:
        count_category = float(dt_kind[str_category])
        probability = count_category/count  # 此分类的个数
        log_probability = -math.log(probability, 2)
        sum_entropy += probability * log_probability
    return sum_entropy


def choose_best_feather_split(a_data_set: List[Tuple]) ->int:
    """
    从a_data_set中找到最好的划分
    实现:
    对每一个属性p的每个值从a_data_set中去除,得到一个熵
    然后对这个属性p的各个熵值做类似期望计算(这一步不是很了解)
    计算得到一个熵值s
    对比a_data_set的原始熵值o
    取o-s为最大值的那个属性
    原理:
    哪一个属性最显著则他拥有的熵是最大的
    @param a_data_set:
    @return:
    """
    base_entropy = calculate_shannon_entropy(a_data_set)
    best_entropy_index = -1
    entropy_gain = 0.0
    count_split = get_property_count(a_data_set)
    for i in range(0, count_split):
        entropy = 0.0
        for current_split_value in get_property_values(a_data_set, i):
            sub_data_set = split_data_set(a_data_set, i, current_split_value)
            sub_entropy = calculate_shannon_entropy(sub_data_set)
            entropy += len(sub_data_set)/double(len(a_data_set))*sub_entropy
        if base_entropy - entropy > entropy_gain:
            entropy_gain = base_entropy - entropy
            best_entropy_index = i
    return best_entropy_index


def create_decision_tree(a_data_set: List[Tuple], a_property_names: Tuple):
    """
    创建决策树
    完全根据理解书写的函数
    最后结果与预期一致!
    实现:
    每次都对data_set计算用哪一个Property划分
    随后划分为sub_data_set[]
    如果一个sub_data_set的所有分类都已经一致,说明这个分支可以结束分类
    如果一个sub_data_set只剩下一个Property(而且里面可能拥有几个分类),则计算分类的概率,取概率最大的分类为这个分支的结果
    如果一个sub_data_set拥有2个以上(包括2个)Property则继续递归调用create_decision_tree
    @param a_data_set:
    @param a_property_names:
    @return:
    {"Fat":
        {
            {"yes" : "不受欢迎"},
            {"no"  :
                {
                    {"有钱" : "还受点欢迎"},
                    {"没钱" : "不受欢迎"}
                }
            }
        }
    },
    {"Cool": "受欢迎"}
    """
    if len(a_data_set) == 1:
        t = a_data_set[0]
        assert len(a_property_names) == 1, "fuck"
        p = a_property_names[0]
        return dict([p, t[-1]])
    elif len(a_property_names) == 0:
        raise "logical error"
    else:
        bi = choose_best_feather_split(a_data_set)
        best_property_name = a_property_names[bi]
        dt = split_data_set_to_dict(a_data_set, bi)
        # pv: Property Value, vt: splitted_data_set
        for (pv, vt) in dt.items():
            # 计算vt中的分类有多少个
            ls = set([t[-1] for t in vt])
            if len(ls) == 1:
                # vt中的分类个数为1, 说明这个分支分类完成
                dt[pv] = tuple(ls)[0]
            elif len(a_property_names) == 1:
                # 只剩下一个Property(而且里面可能拥有几个分类),则计算分类的概率,取概率最大的分类为这个分支的结果
                dt_value = {}
                total = 0
                label_combo = ""
                for t in vt:
                    dt_value[t[-1]] = dt_value.get(t[-1], 0) + 1
                    total += 1
                for (k, v) in dt_value.items():
                    label = "%s(%f) " % (k, v/float(total))
                    label_combo += label
                dt[pv] = {a_property_names[0]: label_combo}
            else:
                dt[pv] = create_decision_tree(vt, remove_tuple_item(a_property_names, bi))
        return {best_property_name: dt}


def remove_tuple_item(a_tuple: Tuple, a_index: int):
    vt = list(a_tuple)
    vt.pop(a_index)
    return tuple(vt)


def split_data_set(a_data_set: List[Tuple], a_property_index: int, a_property_value: int)->List[List[Tuple]]:
    """
    创建新的数据集
    从原数据集中剔除第i个属性, 且其此属性的值为 a_value
    @param a_data_set:
        原数据集,不做修改
    @param a_property_index:
        属性的序列号
    @param a_property_value:
        属性的值
    @return:
        [[1, 1, yes], [1, 1, yes], [1, 0, no], [0, 1, no], [0, 0, no]]
        split_data_set(datas, 0, 1)
        [[1, yes], [1, yes], [0, no]]
    """
    vt = []
    for t in a_data_set:
        value = t[a_property_index]
        if value == a_property_value:
            list_t = list(t)
            list_t.pop(a_property_index)
            vt.append(tuple(list_t))
    return vt


def split_data_set_to_dict(a_data_set: List[Tuple], a_property_index: int)->Dict[object, List[Tuple]]:
    """
    根据index个属性值返回dict<one_property_value, splitted_data_set>
    每一个key是第index个Property的值
    每一个value是第index个Property为key值时的 splitted_data_set
    @param a_data_set:
    @param a_property_index:
    @return:
    """
    dt = {}
    property_value = get_property_values(a_data_set, a_property_index)
    for v in property_value:
        dt[v] = []
    for t in a_data_set:
        dt[t[a_property_index]].append(remove_tuple_item(t, a_property_index))
    return dt


def get_property_count(a_data_set: List[Tuple]) -> int:
    """
    返回数据集属性的个数
    @param a_data_set:
    @return:
    """
    if len(a_data_set) == 0:
        return 0
    return len(a_data_set[0]) - 1


def get_property_values(a_data_set: List[Tuple], a_property_index: int) -> List[int]:
    """
    获取数据集中某一个属性的全部值
    @param a_data_set:
    @param a_property_index:
    @return:返回数据集中某一个属性的全部值
    """
    dt = {}
    for item in a_data_set:
        property_value = item[a_property_index]
        dt[property_value] = 1
    return dt.keys()










