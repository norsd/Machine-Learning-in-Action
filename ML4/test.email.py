# coding:utf-8

from typing import List
from typing import Tuple
from typing import TypeVar

import re
import os
import ML4.NativeBayes
import ML4.NativeBayes2

__author__ = 'di_shen_sh@gmail.com'

# 从datas/email/文件夹下分别读取2个已经分类的文件夹

def parse_text(a_text:str)->List[str]:
    tokens = re.split(r'\W+', a_text)
    return [w.lower() for w in tokens if len(w)>2]

script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)

sample_data_dirs = [script_dir + "/datas/email/ham", script_dir + "/datas/email/spam"]
sample_data_labels = [0, 1]

datas_set = []
labels_set = []

for i in range(0, len(sample_data_labels)):
    label = sample_data_labels[i]
    data_dir = sample_data_dirs[i]
    file_names_ok = os.listdir(data_dir)
    for j in range(0, 20):
        name = file_names_ok[j]
        file = open(data_dir + "/" + name)
        text = file.read()
        datas = parse_text(text)
        datas_set.append(datas)
        labels_set.append(label)


bayes = ML4.NativeBayes2.NativeBayes2()
bayes.add_samples(datas_set, labels_set)


test_data_dirs = [script_dir + "/datas/email/ham", script_dir + "/datas/email/spam"]
test_data_correct_labels = [0, 1]

test_count = 0
test_correct_count = 0
for i in range(0, len(test_data_correct_labels)):
    label = test_data_correct_labels[i]
    data_dir = test_data_dirs[i]
    file_names_ok = os.listdir(data_dir)
    for j in range(21, 25):
        test_count += 1
        name = file_names_ok[j]
        file = open(data_dir + "/" + name)
        text = file.read()
        datas = parse_text(text)
        r = bayes.test(datas)
        print(j)
        print(r)
        cauculate_label = r[1]
        if label == cauculate_label:
            test_correct_count += 1

print("Error Rate:{0}".format(test_correct_count/test_count))