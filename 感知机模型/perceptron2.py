import numpy as np
import random
import matplotlib.pyplot as plt
import math
x = np.loadtxt(open("ex4x.dat", "rb"), skiprows=0)
y = np.loadtxt(open("ex4y.dat", "rb"), skiprows=0)
x0 = np.ones((80, 1), float)
x = np.c_[x0, x]

for yi in range(40, 80):
    y[yi] = 2

theta1 = np.array([-10.0, 10.0, 0.1])  # 录取，y[i]=1
theta2 = np.array([10.0, -1.4, 10.2])  # 未录取，y[i]=0

study_gsd = 0.0001


def h_theta(ii, sort):
    if sort == 1:
        return np.dot(x[ii, 0:3], theta1)  # 录取
    else:
        return np.dot(x[ii, 0:3], theta2)  # 未录取


def argmax(ii):
    if np.dot(x[ii, 0:3], theta1) > np.dot(x[ii, 0:3], theta2):
        return 1
    else:
        return 2


def loss():
    tt = 0.0
    for ii in range(80):
        if argmax(ii) != y[ii]:
            tt = tt + (h_theta(ii, argmax(ii)) - h_theta(ii, y[ii]))
    return tt / 80


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
    t_theta0 = theta1[0] - theta2[0]
    t_theta1 = theta1[1] - theta2[1]
    t_theta2 = theta1[2] - theta2[2]
    X2 = - (t_theta0 + t_theta1 * X1) / t_theta2
    plt.plot(X1, X2)
    plt.show()


if __name__ == '__main__':
    for i in range(200000):
        m = random.randint(0, 79)

        # 对theta1
        gradient = 0.0
        c = argmax(m)
        if c != y[m]:
            if c == 1 :
                gradient = x[m]
            if y[m] == 1:
                gradient = -x[m]
        t_theta = theta1 - study_gsd * gradient

        # 对theta2
        gradient = 0.0
        c = argmax(m)
        if c != y[m]:
            if c == 2:
                gradient = x[m]
            if y[m] == 2:
                gradient = -x[m]
        theta2 = theta2 - study_gsd * gradient

        theta1 = t_theta
        print(loss(), i)
    print(theta1, theta2)
    draw()
