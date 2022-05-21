import numpy as np

x = np.loadtxt(open("ex4x.dat", "rb"), skiprows=0)
y = np.loadtxt(open("ex4y.dat", "rb"), skiprows=0)

w1 = np.random.randint(0, 1, (2, 5)).astype(np.float64)

w2 = np.random.randint(0, 1, (5, 2)).astype(np.float64)

b1 = np.random.randint(1, 2, (1, 5)).astype(np.float64)

b2 = np.random.randint(1, 2, (1, 2)).astype(np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


study = 0.001

if __name__ == '__main__':
    for iii in range(10000):
        for i in range(80):
            z1 = np.dot(x[i, :], w1) + b1  # 1x5
            a1 = 1.0 / (1.0 + np.exp(-z1))  # 1x5
            z2 = np.dot(a1, w2) + b2  # 1x2
            a2 = 1.0 / (1.0 + np.exp(-z2))  # 1x2

            # yy = [0.0, 0.0]
            if y[i] == 1:
                yy = [1.0, 0.0]
            else:
                yy = [0.0, 1.0]

            error2 = a2 * (1 - a2) * (a2 - yy)  # 1x2
            temp = a1 * (1 - a1)
            diag = np.diag([temp[0][k] for k in range(5)])
            error1 = np.dot(np.dot(diag, w2), error2.T)  # 5x1

            w2 = w2 - study * np.dot(a1.T, error2)
            b2 = b2 - study * error2
            w1 = w1 - study * np.dot(error1, np.array([x[i, 0:2]])).T
            b1 = b1 - study * error1.T

            tz1 = np.dot(x[i, :], w1) + b1
            ta1 = 1.0 / (1.0 + np.exp(-tz1))
            z2 = np.dot(ta1, w2) + b2
            ta2 = 1.0 / (1.0 + np.exp(-z2))  # 1x2

            loss = np.sum((ta2 - yy) ** 2)
            print(loss)

    k = 0
    rate = 0.0
    for ii in range(5):
        count = 0.0  # 预测对了的个数
        for i in range(k, k + 16):

            tz1 = np.dot(x[i, :], w1) + b1
            ta1 = 1.0 / (1.0 + np.exp(-tz1))
            z2 = np.dot(ta1, w2) + b2
            ta2 = 1.0 / (1.0 + np.exp(-z2))  # 1x2

            if y[i] == 1:
                if ta2[0, 0] > ta2[0, 1]:
                    count += 1
            else:
                if ta2[0, 0] < ta2[0, 1]:
                    count += 1
        k = k + 16
        print("第", ii + 1, "个子集正确个数:", count)
        rate = rate + count / 16
    print("正确率:", rate/5)
