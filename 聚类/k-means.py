import numpy as np
import matplotlib.pyplot as plt


x = np.loadtxt(open("ex4x.dat", "rb"), skiprows=0)
y = np.ones((80, 1), float)

k = np.array([[1.0, 80.0], [0.0, 1.0]])


def draw():
    plt.xlim(xmax=65, xmin=15)
    plt.ylim(ymax=90, ymin=40)

    color = '#DC143C'

    color1 = '#0000FF'
    color2 = '#008000'
    for ii in range(80):
        if y[ii] == 0:
            plt.scatter(x[ii, 0], x[ii, 1], c=color1)
        else:
            plt.scatter(x[ii, 0], x[ii, 1], c=color2)

    plt.scatter(k[0, 0], k[0, 1], c=color, s=200)
    plt.scatter(k[1, 0], k[1, 1], c=color, s=200)

    plt.show()


if __name__ == '__main__':

    for i in range(1000):
        # 为点赋类
        for j in range(80):
            t0 = (x[j, 0] - k[0, 0]) ** 2 + (x[j, 1] - k[0, 1]) ** 2
            t1 = (x[j, 0] - k[1, 0]) ** 2 + (x[j, 1] - k[1, 1]) ** 2
            if t0 > t1:
                y[j] = 0
            else:
                y[j] = 1

        # 调整中心
        x0 = np.array([0.0, 0.0])
        count0 = 0
        x1 = np.array([0.0, 0.0])
        count1 = 0
        for m in range(80):
            if y[m] == 0:
                x0 += x[m]
                count0 += 1
            else:
                x1 += x[m]
                count1 += 1
        k[0] = x0 / count0
        k[1] = x1 / count1

    draw()
