import numpy as np
import random
import matplotlib.pyplot as plt
import math
x = np.loadtxt(open("ex4x.dat", "rb"), skiprows=0)
y = np.loadtxt(open("ex4y.dat", "rb"), skiprows=0)
x0 = np.ones((80, 1), float)
x = np.c_[x0, x]

for yi in range(40, 80):
    y[yi] = -1

theta = np.array([-10.0, 10, 0.1])

study_gsd = 0.0001


def h_theta(ii):
    return np.dot(x[ii, 0:3], theta)


def loss():
    t = 0.0
    for i in range(80):
        t = t + max(0, -y[i] * h_theta(i))
    return t / 80


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
    X2 = - (theta[0] + theta[1] * X1) / theta[2]
    plt.plot(X1, X2)
    plt.show()


if __name__ == '__main__':
    for i in range(200000):
        t = np.array([0.0, 0.0, 0.0])
        j = random.randint(0, 79)

        if y[j] * h_theta(j) < 0:
            t = - y[j] * x[j]
        theta = theta - study_gsd * t
        print(loss(), i)
    print(theta)
    draw()
