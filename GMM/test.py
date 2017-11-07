import numpy as np
from matplotlib import pyplot as plt
import random

def creatDate(nx):
    x = np.random.rand(nx, 1) * 10 + np.array([[1, 1]])
    for i in range((int)(np.random.rand(1)[0] * 10) + 40):
        x = np.hstack((x, np.random.rand(nx, 1) * 5 + np.array([[1], [1]])))

    for i in range((int)(np.random.rand(1)[0] * 10) + 40):
        x = np.hstack((x, np.random.rand(nx, 1) * 5 + np.array([[25], [25]])))

    for i in range((int)(np.random.rand(1)[0] * 10) + 40):
        x = np.hstack((x, np.random.rand(nx, 1) * 5 + np.array([[50], [1]])))
    # print(x.shape)
    return x

def readFile(filename=''):
    '''
    读取数据
    :param filename:
    :return:
    '''
    label = []
    x = []
    with open(filename) as file:
        for eachLine in file:
            eachLineData = eachLine.rstrip().split(' ')
            label.append(int(eachLineData[0]))
            x.append([float(e) for e in eachLineData[1:]])
    label = np.array(label)
    label = label.reshape(label.shape[0], 1).T
    x = np.array(x).T
    return label, x

def getCircle(mu, sigma):
    points = 100

    center = mu
    radius = np.linalg.det(sigma)

    tt = np.array([np.linspace(0, 2 * np.pi, points)])  # 产生的点
    x = center[0, :] + np.cos(tt) * radius
    y = center[1, :] + np.sin(tt) * radius
    return x, y

def init(x):
    n, m = x.shape

    # 初始化均值
    mu = np.mean(x, axis=1, keepdims=True)

    # 初始化seita
    s = np.cov(x)
    u, simga, vT = np.linalg.svd(s)
    beita = 0.5 * np.max(simga)
    seita = beita * ((4 / ((n + 2) * m)) ** (1 / (n + 4)))

    # 计算phi矩阵
    phi = np.zeros((m, m))
    for n1 in range(m):
        for n2 in range(n1 + 1):
            xi = x[:, n1].reshape(n, 1)
            xj = x[:, n2].reshape(n, 1)
            temp = (2 * np.pi * seita ** 2) ** (- n / 2) * np.exp(-0.5 * np.dot((xi - xj).T, xi - xj) / seita ** 2)
            phi[n1, n2] = temp
            phi[n2, n1] = temp

    # 初始化s
    s = seita * np.identity(n)

    # 初始化alpha
    alpha = 0.5

    # 初始化mixGau
    mixGau = [np.zeros((1, m))]
    for j in range(m):
        mixGau[0][0, j] = alpha * gaussian(x[:, j].reshape(n, 1), mu, s)

    # 计算pM
    pM = np.zeros((1, m))
    for j in range(m):
        pM[0, j] = temp / ((1 - alpha) * mixGau[0][0, j] + temp)

    l = 0
    for j in range(m):
        temp = alpha * gaussian(x[:, j].reshape(n, 1), mu, s)
        l = l + np.log(alpha * gaussian(x[:, j].reshape(n, 1), mu, s))

    return mu, seita, phi, s, alpha, mixGau, pM, l

def selectParameters(k, x, mixGau, phi, seita):
    k = k - 1
    n, m = x.shape

    # 求初始均值
    maxNearLoss = -1e10
    muLocat = 0
    for j in range(m):
        temp = nearLoss(k - 1, j, x, mixGau, phi)
        if maxNearLoss < temp:
            maxNearLoss = temp
            muLocat = j
    mu = x[muLocat, :].reshape(1, m)

    # 求初始协方差
    s = seita * np.identity(n)

    # 求初始参数alpha
    alpha = 0.5 - 0.5 * np.sum(lossDer(k, muLocat, mixGau, phi)) / (np.sum((lossDer(k, muLocat, mixGau, phi)) ** 2))
    if alpha <= 0 or alpha >= 1:
        if k == 1:
            alpha = 0.5
        else:
            alpha = 2 / (k + 1)

    return mu, s, alpha

def globalEM(x):
    n, m = x.shape
    k = 0
    mu, seita, phi, s, alpha, mixGau, pM, l = init(x)

    parameters = []
    parameters.append({'mu': mu, 's': s, 'alpha': alpha})

    newL = l

    while True:
        l = newL
        k = k + 1
        mu, s, alpha = selectParameters(k, x, mixGau, phi, seita)
        mu, s, alpha, pM = partialEM(k, x, pM, mu, s, alpha, mixGau)
        newL = loss(k, x, mu, s, mixGau, alpha)
        if newL <= l:
            break


def partialEM(k, x, pM, mu, s, alpha, mixGau):
    k = k - 1
    n, m = x.shape
    l = loss(k, x, mu, s, mixGau, alpha)
    oldL = l / 2
    newPM = np.zeros((1, m))

    while np.abs(l / oldL - 1) > 1e-6:
        oldL = l

        for j in range(m):
            temp = alpha * gaussian(x[:, j].reshape(n, 1), mu, s)
            newPM[0, j] = temp / ((1 - alpha) * mixGau[k][0, j] + temp)
        newPMSum = np.sum(newPM)

        # 更新参数
        alpha = 1 / m * newPMSum
        mu = np.sum(newPM * x, axis=1, keepdims=True) / newPMSum

        sMol = 0
        for j in range(m):
            sMol = sMol + newPM[0, j] * np.dot((x[:, j].reshape(n, 1) - mu).T, x[:, j].reshape(n, 1) - mu)
        s = sMol / newPMSum

        l = loss(k, x, mu, s, mixGau, alpha)

    return mu, s, alpha, np.vstack((pM, newPM))

def loss(k, x, mu, s, mixGau, alpha):
    n, m = x.shape
    k = k - 1

    l = 0
    for j in range(m):
        l = l + np.log((1 - alpha) * mixGau[k][0, j] + alpha * gaussian(x, mu, s))
    return l

def nearLoss(k, muLocat, x, mixGau, phi):
    k = k - 1
    partOne = np.sum(np.log((mixGau[k] + phi[muLocat, :].reshape(1, x.shape[1])) / 2))
    partTwo = 0.5 * (np.sum(lossDer(k, muLocat, mixGau, phi))) ** 2 / (np.sum((lossDer(k, muLocat, mixGau, phi)) ** 2))
    return partOne + partTwo

def lossDer(k, muLocat, mixGau, phi):
    k = k - 1
    return (mixGau[k] - phi[muLocat, :].reshape(1, phi.shape[0])) / (mixGau[k] + phi[muLocat, :].reshape(1, phi.shape[0]))


def gaussian(xj, mu, s):
    n, m = xj.shape

    a = 1.0 / ((2 * np.pi) ** (n / 2) * np.linalg.det(s) ** 0.5)
    b = - 0.5 * np.dot(np.dot((xj - mu).T, np.linalg.inv(s)), xj - mu)
    p = a * np.exp(b)
    return p


if __name__ == '__main__':
    # label, x = readFile('data.txt')
    x = creatDate(2)
    globalEM(x)





    # C, parameters = gmm(x, 3, 10000, 0.00001)
    # print(C)

    # plt.figure(1)
    # plt.subplot(1, 2, 1)
    # plt.grid()
    # plt.plot(x[0, :], x[1, :], '+')
    # plt.subplot(1, 2, 2)
    # plt.grid()
    # for i in C[0]:
    #     plt.plot(x[0, i], x[1, i], 'r*')
    # cx, cy = getCircle(parameters['muList'][0], parameters['sigmaList'][0])
    # plt.plot(cx[0,:], cy[0,:], 'r')
    #
    # for i in C[1]:
    #     plt.plot(x[0, i], x[1, i], 'b^')
    # cx, cy = getCircle(parameters['muList'][1], parameters['sigmaList'][1])
    # plt.plot(cx[0,:], cy[0,:], 'b')
    #
    # for i in C[2]:
    #     plt.plot(x[0, i], x[1, i], 'g+')
    # cx, cy = getCircle(parameters['muList'][2], parameters['sigmaList'][2])
    # plt.plot(cx[0,:], cy[0,:], 'g')
    #
    # plt.show()

#
#
#
# if __name__ == '__main__':
#     # label, x = readFile('data.txt')
#     x = creatDate(2)
#     S = np.cov(x)
#     print(S)
#     print(S.shape)
#     U, Simga, VT = np.linalg.svd(S)
#     print(Simga)
#     beita = 0.5 * np.max(Simga)
#     print(beita)
#     n, m = x.shape
#     seita = beita * ((4 / ((n + 2) * m)) ** (1 / (n + 4)))
#     print(seita)