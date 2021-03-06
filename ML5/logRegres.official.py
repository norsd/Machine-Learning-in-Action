__author__ = 'di_shen_sh@163.com'

import os

from numpy import *

def loadDataSet(a_sampleFilePath):
    print(a_sampleFilePath)
    dataMat = []; labelMat = []
    fr = open(a_sampleFilePath)
    for line in fr.readlines():
        cols = line.strip().split()
        dataMat.append([1.0, float(cols[0]), float(cols[1])])
        labelMat.append(int(cols[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels, vtWeights = [] ):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    del vtWeights[:]
    for i in range(n):
        vtWeights.append([])
    alpha = 0.001
    maxCycles = 500
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()*error
        for i in range(n):
            vtWeights[i].append(array(weights)[i][0])
    print(shape(error))
    print(shape(dataMatrix.transpose()))
    return weights

def plotBestFit(dataMat, labelMat):
    import matplotlib.pyplot as plt
    #dataMat, labelMat= loadDataSet()
    weights = gradAscent(dataMat, labelMat).getA()
    datass = array(dataMat)
    n = shape(datass)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(datass[i, 1]); ycord1.append(datass[i, 2])
        else:
            xcord2.append(datass[i, 1]); ycord2.append(datass[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    print(weights)
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


# Test
script_path = os.path.realpath(__file__)
script_dir = os.path.dirname(script_path)
dataArr, labelMat = loadDataSet("{0}/{1}".format(script_dir, "testSet.txt"))
print(gradAscent(dataArr, labelMat))