import numpy as np
import random
import math
import matplotlib.pyplot as plt
x = np.loadtxt(open("ex4x.dat", "rb"), skiprows=0)
y = np.loadtxt(open("ex4y.dat", "rb"), skiprows=0)
x0 = np.ones((80, 1), float)
x = np.c_[x0, x]
theta0 = np.array([-1.0, 0.1, 0.1])
theta1 = np.array([-1.0, 0.1, 0.1])
#[-16.38, 0.1483, 0.1589]

study = 0.0007
study_gsd = 0.00001


def softmax(i, j):
    t0 = math.exp(np.dot(x[i, 0:3], theta0))
    t1 = math.exp(np.dot(x[i, 0:3], theta1))
    if j == 0:
        return t0 / (t0 + t1)
    else:
        return t1 / (t0 + t1)


def loss():
    tt = 0.0
    for ii in range(80):
        if y[ii] == 0:
            tt = tt + math.log(softmax(ii, 0))
        else:
            tt = tt + math.log(softmax(ii, 1))

    return - tt / 80


def mse():
    tm = 0.0
    for ii in range(0, 40):
        tm = tm + (1 - softmax(ii, 1)) ** 2
    for ii in range(40, 80):
        tm = tm + (0 - softmax(ii, 0)) ** 2
    return tm / 80


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
    t_theta0 = theta1[0] - theta0[0]
    t_theta1 = theta1[1] - theta0[1]
    t_theta2 = theta1[2] - theta0[2]
    X2 = - (t_theta0 + t_theta1 * X1) / t_theta2
    plt.plot(X1, X2)
    plt.show()


if __name__ == '__main__':


    "梯度下降"

    for i in range(100000):

        t = np.array([0.0, 0.0, 0.0])
        for j in range(80):
            if y[j] == 0:
                t = t + (1.0 - softmax(j, 0)) * x[j, 0:3]
            else:
                t = t + (0.0 - softmax(j, 0)) * x[j, 0:3]
        t_theta = theta0 + study * t / 80

        t = np.array([0.0, 0.0, 0.0])
        for j in range(80):
            if y[j] == 1:
                t = t + (1.0 - softmax(j, 1)) * x[j, 0:3]
            else:
                t = t + (0.0 - softmax(j, 1)) * x[j, 0:3]
        theta1 = theta1 + study * t / 80

        theta0 = t_theta

        print(loss(), i)
    print(theta0, theta1)
    draw()


    "随机梯度下降"

    # for i in range(1000000):
    #     j = random.randint(0, 79)
    #     if y[j] == 0:
    #         theta0 = theta0 + study_gsd * (1 - softmax(j, 0)) * x[j, 0:3]
    #         theta1 = theta1 + study_gsd * (0 - softmax(j, 1)) * x[j, 0:3]
    #     else:
    #         theta0 = theta0 + study_gsd * (0 - softmax(j, 0)) * x[j, 0:3]
    #         theta1 = theta1 + study_gsd * (1 - softmax(j, 1)) * x[j, 0:3]
    #     print(loss(), i)
    # print(theta0, theta1)
    # # draw()

