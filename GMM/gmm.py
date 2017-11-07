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

def get_circle(mu, sigma):
    points = 100

    center = mu
    radius = np.linalg.det(sigma)

    tt = np.array([np.linspace(0, 2 * np.pi, points)])  # 产生的点
    x = center[0, :] + np.cos(tt) * radius
    y = center[1, :] + np.sin(tt) * radius
    return x, y

def gaussian(x, mu, sigma):
    '''
    高斯概率密度函数
    :param x: (n, 1)
    :return: p:
    '''
    # m = x.shape[1]
    n = x.shape[0]

    a = 1.0 / ((2 * np.pi) ** (n / 2) * np.linalg.det(sigma) ** 0.5)
    b = - 0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), x - mu)
    p = a * np.exp(b)
    return p

def mixGaussian(xj, ki, parameters):
    alpha = parameters['alpha']
    k = len(alpha)

    pM = alpha[ki] * gaussian(xj, parameters['muList'][ki], parameters['sigmaList'][ki])
    pMSum = 1e-20 # 1e-20防止pMSum为0
    for l in range(k):
        pMSum = pMSum + alpha[l] * gaussian(xj, parameters['muList'][l], parameters['sigmaList'][l])
    # if pMSum == 0 or pMSum == np.nan:
    #     for l in range(k):
    #         print(alpha[l] * gaussian(xj, parameters['muList'][l], parameters['sigmaList'][l]))
    pM = pM / pMSum

    return pM

# def cost(gamma, limit):
#     m = gamma.shape[1]
#     for i in range(m):
#         if np.max(gamma[:, i]) < limit and np.min(gamma[:, i]) > (1 - limit):
#             return False
#     return True

def gmm(x, k, iteration=1000, disMu=0.001):
    n, m = x.shape

    # 初始化
    parameters = {'alpha': [1.0 / k for i in range(k)],
                  # 'muList': [np.array([[1],[1]]), np.array([[25],[25]]), np.array([[50],[1]])],
                  'muList': [x[:, i].reshape(n, 1) for i in random.sample(range(m), k)],
                  'sigmaList': [np.identity(n) * 0.1 for i in range(k)]
                  }
    gamma = np.zeros((k, m))
    it = 0
    dMu = 10000
    while it <= iteration and dMu > disMu:
        print('Iteration: ' + str(it))
        # 计算后验概率
        for i in range(k):
            for j in range(m):
                gamma[i, j] = mixGaussian(x[:, j].reshape(n, 1), i, parameters)

        # 更新参数
        sumGamma = np.sum(gamma, axis=1, keepdims=True) + 1e-20
        oldMu = parameters['muList'].copy()
        dMu = 0
        for i in range(k):
            newMu = np.sum(x * gamma[i, :], axis=1, keepdims=True) / sumGamma[i, 0]
            parameters['muList'][i] = newMu
            parameters['sigmaList'][i] = np.dot(gamma[i, :] * (x - newMu), (x - newMu).T) / sumGamma[i, 0]
            parameters['alpha'][i] = sumGamma[i, 0] / m
            dMu = dMu + np.sum(np.abs(oldMu[i] - newMu))
            # print(parameters['sigmaList'][i].shape)
        it = it + 1
    print('Iteration: ' + str(it))
    C = {}
    for j in range(k):
        C[j] = []

    for i in range(m):
        C[np.argwhere(gamma[:, i] == np.max(gamma[:, i]))[0,0]].append(i)
    return C, parameters

if __name__ == '__main__':
    # label, x = readFile('data.txt')
    x = creatDate(2)
    C, parameters = gmm(x, 3, 10000, 0.00001)
    # print(C)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.plot(x[0, :], x[1, :], '+')
    plt.subplot(1, 2, 2)
    plt.grid()
    for i in C[0]:
        plt.plot(x[0, i], x[1, i], 'r*')
    cx, cy = get_circle(parameters['muList'][0], parameters['sigmaList'][0])
    plt.plot(cx[0,:], cy[0,:], 'r')

    for i in C[1]:
        plt.plot(x[0, i], x[1, i], 'b^')
    cx, cy = get_circle(parameters['muList'][1], parameters['sigmaList'][1])
    plt.plot(cx[0,:], cy[0,:], 'b')

    for i in C[2]:
        plt.plot(x[0, i], x[1, i], 'g+')
    cx, cy = get_circle(parameters['muList'][2], parameters['sigmaList'][2])
    plt.plot(cx[0,:], cy[0,:], 'g')

    plt.show()

