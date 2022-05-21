import numpy as np
import random
import matplotlib.pyplot as plt
import math
x = np.loadtxt(open("ex4x.dat", "rb"), skiprows=0)
y = np.loadtxt(open("ex4y.dat", "rb"), skiprows=0)
x0 = np.ones((80, 1), float)
x = np.c_[x0, x]
seita = np.array([-1.0, 0.1, 0.1])
#[-16.38, 0.1483, 0.1589]

study = 0.0014
study_gsd = 0.0001


def h_seita(i):
    return 1.0/(1.0 + math.exp(- np.dot(x[i, 0:3], seita)))


def draw():
    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.xlim(xmax=65, xmin=15)
    plt.ylim(ymax=90, ymin=40)

    area = np.pi * 4 ** 2
    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#DC143C'

    plt.scatter(x[0:40, 1], x[0:40, 2], s=area, c=colors1, alpha=0.4, label='类别A')
    plt.scatter(x[40:80, 1], x[40:80, 2], s=area, c=colors2, alpha=0.4, label='类别B')
    X1 = x[0:80, 1]
    X2 = - (seita[0] + seita[1] * X1) / seita[2]
    plt.plot(X1, X2)
    plt.show()


if __name__ == '__main__':

    "随机梯度下降"

    for i in range(1000000):
        t = np.array([0.0, 0.0, 0.0])
        j = random.randint(0, 79)
        t = (y[j] - h_seita(j)) * x[j, 0:3]
        seita = seita + study_gsd * t
        print(y[j] * math.log(h_seita(j)) - (1 - y[j]) * math.log(1 - h_seita(j)), i)
    print(seita)
    draw()


