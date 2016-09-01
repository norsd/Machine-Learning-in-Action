__author__ = 'di_shen_sh@163.com'

import os
import ML5.GradientDescent
from numpy import *

def loadDataSet(a_sampleFilePath):
    print(a_sampleFilePath)
    dataMat = []; labelMat = []
    fr = open(a_sampleFilePath)
    for line in fr.readlines():
        cols = line.strip().split()
        dataMat.append([float(cols[0]), float(cols[1])])
        labelMat.append(int(cols[2]))
    return dataMat, labelMat

# Test
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)
dataArr, labelMat = loadDataSet("{0}/{1}".format(script_dir, "testSet.txt"))

gd = ML5.GradientDescent.GradientDescent()
print(gd.add_samples(dataArr, labelMat))
print(gd.calculate(0.001, 5000))
gd.draw()