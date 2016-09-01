# coding:utf-8
from numpy import *
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypeVar

__auth__ = 'di_shen_sh@gmail.com'

T = TypeVar('T')


class GradientDescent:
    # m 代表样本个数
    # n 代表θ个数
    θn1 = zeros(0)
    xmn = zeros(0)
    ym1 = zeros(0)

    def __init__(self):
        dt_dt_label = {}
        return

    def __sigmoid(self, a_value:float):
        return 1.0/(1+exp(-a_value))

    def add_samples(self, a_datas, a_labels):
        self.xmn = mat(a_datas)
        m,n = shape(self.xmn)
        self.xmn = insert(self.xmn, 0, values = ones(m), axis=1)
        self.ym1 = mat(a_labels).transpose()
        if shape(self.ym1)[0] != m:
            return False
        n += 1
        self.θn1 = ones((n,1))
        return True

    def calculate(self, a_step:float, a_count:int):
        step = a_step # 0.000001
        for i in range(0, a_count):
            # xm: (m,n)
            # θv: (n,1)
            # deltas: (m,1)
            Δm1 = self.__sigmoid(self.xmn * self.θn1) - self.ym1
            xnm = self.xmn.transpose()
            rn1 = xnm * Δm1 * step
            self.θn1 = self.θn1 - rn1
        return self.θn1

    def draw(self):
        import matplotlib.pyplot as plt
        x0s_0 = []
        x1s_0 = []
        x0s_1 = []
        x1s_1 = []
        m,n = shape(self.xmn)
        if n>3:
            print("无法绘画{0}维数据".format(n))
            return
        for i in range(0, m):
            if self.ym1[i] == 0:
                x0s_0.append(self.xmn[i,1]); x1s_0.append(self.xmn[i,2])
            else:
                x0s_1.append(self.xmn[i,1]); x1s_1.append(self.xmn[i,2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x0s_0, x1s_0, s=30, c='red', marker='s')
        ax.scatter(x0s_1, x1s_1, s=30, c='green')
        x = arange(-3.0, 3.0, 0.1)
        y = -(self.θn1[0] + self.θn1[1]*x)/self.θn1[2]
        y = y.transpose()
        print(shape(x))
        print(shape(y))
        ax.plot(x,y)
        plt.show()
