import numpy as np
from creatData import *
import matplotlib.pyplot as plt
import copy
import math

class GGM2:

    def __init__(self):
        '''
        D: (n, m)维ndarray，其中n是样本x的属性数量，m是样本的总数
        params: list，每个元素中储存{'mu': ..., 's': ..., 'alpha': ...}
        age: 世代数，也表示k个类别
        '''

        self.D = creatDate(2)
        self.n, self.m = self.D.shape
        self.age = 0
        self.iterLimit = 100

        # 计算seita
        u, simga, vT = np.linalg.svd(np.cov(self.D))
        beita = 0.5 * np.max(simga)
        self.seita = beita * ((4 / ((self.n + 2) * self.m)) ** (1 / (self.n + 4)))

        # 计算K_{ij}矩阵，称为phi
        self.phi = np.zeros((self.m, self.m))
        for n1 in range(self.m):
            for n2 in range(n1 + 1):
                xi = self.D[:, n1].reshape(self.n, 1)
                xj = self.D[:, n2].reshape(self.n, 1)
                temp = (2 * np.pi * self.seita ** 2) ** (- self.n / 2) * np.exp(
                    -0.5 * np.dot((xi - xj).T, xi - xj) / self.seita ** 2)
                self.phi[n1, n2] = temp
                self.phi[n2, n1] = temp


    def globalEM(self):
        it = 0
        # 1. 初始化参数
        params = self.init()
        l = self.loss(params)


        while self.age < self.iterLimit:
            # plt.close(self.age) #############
            self.age = self.age + 1
            #
            # plt.figure(self.age)  #############
            # plt.plot(self.D[0, :], self.D[1, :], 'b+')  ################
            # plt.ion()  ################
            # plt.plot(params['mu'][0][0,:], params['mu'][0][0,:], 'r*')

            oldL = l
            oldParams = copy.deepcopy(params)

            # 2. 选择初始化参数
            self.selectParam(params)
            # l = self.loss(params)
            # if l <= oldL:
            #     return oldL, oldParams

            # 3. 部分EM算法
            l = self.partialEM(params)

            # for i in range(len(params['mu'])):
            #     plt.plot(params['mu'][i][0, 0], params['mu'][i][1, 0], 'r*')

            if l <= oldL:
                return oldL, oldParams
        return l, params

    def init(self):
        n, m = self.D.shape

        # 计算初始的mu值
        mu = [np.mean(self.D, axis=1, keepdims=True)]

        # 初始化seita和s
        s = [self.seita ** 2 * np.identity(n)]

        # 初始化alpha
        alpha = [0.5]
        # alpha = [1]

        params = {'mu': mu, 's': s, 'alpha': alpha}

        return params

    def selectParam(self, params):
        k = len(params['mu'])

        # 选取初始mu
        f = np.zeros((1, self.m))
        for j in range(self.m):
            for i in range(k):
                f[0, j] = f[0, j] + params['alpha'][i] * self.gaussian(self.D[:, j].reshape(self.n, 1), params['mu'][i], params['s'][i])
        muLocat = 0
        l = -1e100
        # lList = [] ######################
        for locat in range(self.m):
            oldL = l
            temp = ((f[0, :] - self.phi[locat, :]) / (f[0, :] + self.phi[locat, :])).reshape(1, self.m)
            partOne = np.sum(np.log((f[0, :] + self.phi[locat, :]) / 2).reshape(1, self.m))
            partTwo = 0.5 * (np.sum(temp)) ** 2 / np.sum(temp ** 2)
            l = partOne + partTwo

            # lList.append(l) ##############

            if l > oldL:
                muLocat = locat

        mu = self.D[:, muLocat].reshape(self.n, 1)
        # plt.plot(mu[0, 0], mu[1, 0], 'r*') #####################

        # 选取初始s
        s = self.seita ** 2 * np.identity(self.n)

        # 选取初始alpha
        temp = ((f[0, :] - self.phi[muLocat, :]) / (f[0, :] + self.phi[muLocat, :])).reshape(1, self.m)
        alpha = 0.5 - 0.5 * np.sum(temp) / np.sum(temp ** 2)
        if alpha <= 0 or alpha >= 1:
            alpha = 2 / (k + 2)

        # 将参数添加进params
        params['mu'].append(mu)
        params['s'].append(s)
        # params['alpha'] = [(1 - alpha) * each for each in params['alpha']]
        params['alpha'].append(alpha)

    def partialEM(self, params):
        k = len(params['mu'])
        l = self.loss(params)

        # lList = [l[0,0]] ##########################
        it = 0
        while it < self.iterLimit:
            # while True:
            it = it + 1
            oldL = l
            gamma = np.zeros((1, self.m))

            # 计算后验概率
            for j in range(self.m):
                gamma[0, j] = self.getGamma(self.D[:, j].reshape(self.n, 1), -1, params)

            # 更新参数
            sumGamma = np.sum(gamma, axis=1, keepdims=True) + 1e-100

            # plt.figure(self.age)####################
            # plt.plot(self.D[0,:], self.D[1,:], 'b+') ####################
            # plt.ion()####################


            newMu = np.sum(self.D * gamma[0, :], axis=1, keepdims=True) / sumGamma[0, 0]
            params['mu'][-1] = newMu
            params['s'][-1] = np.dot(gamma[0, :] * (self.D - newMu), (self.D - newMu).T) / sumGamma[0, 0]
            params['alpha'][-1] = sumGamma[0, 0] / self.m
            # plt.plot(newMu[0,0], newMu[1,0], 'r*') #################

            l = self.loss(params)
            # lList.append(l[0,0]) ######################

            # if l - oldL < -1:
            #     return oldL

            if np.abs(l / oldL - 1) < 10e-6:
                # if np.abs(l / oldL - 1) < 1e-1:
                return l
        return l


    def classifyData(self, params):
        k = len(params['mu'])
        gamma = np.zeros((k, self.m))
        for i in range(k):
            for j in range(self.m):
                gamma[i, j] = self.getGamma(self.D[:, j].reshape(self.n, 1), i, params)

        C = {}
        for i in range(k):
            C[i] = []

        for j in range(self.m):
            C[np.argwhere(gamma[:, j] == np.max(gamma[:, j]))[0, 0]].append(j)
        return C

    def getGamma(self, x, k, params):
        pM = params['alpha'][k] * self.gaussian(x, params['mu'][k], params['s'][k])
        pMSum = 1e-100  # 1e-100防止pMSum为0
        for l in range(len(params['mu']) - 1):
            pMSum = pMSum + params['alpha'][l] * self.gaussian(x, params['mu'][l], params['s'][l])
        pMSum = (1 - params['alpha'][-1]) * pMSum + params['alpha'][-1] * self.gaussian(x, params['mu'][-1], params['s'][-1])
        pM = pM / pMSum

        return pM

    def loss(self, params):
        k = len(params['mu'])
        l = 0
        try:
            for i in range(k):
                temp = 0
                for j in range(self.m):
                    temp = temp + params['alpha'][i] * self.gaussian(self.D[:, j].reshape(self.n, 1), params['mu'][i], params['s'][i])
                try:
                    l = l + np.log(temp)
                except:
                    l = l + -1e100
        except:
            for i in range(k):
                temp = 0
                for j in range(self.m):
                    temp = temp + params['alpha'][i] * self.gaussian(self.D[:, j].reshape(self.n, 1), params['mu'][i], params['s'][i])
                try:
                    l = l + np.log(temp)
                except:
                    l = l + -1e100
            # if temp == 0:
            #     l = l + -1e100
            # else:
            #     l = l + np.log(temp)
        return l

    def gaussian(self, x, mu, s):
        n, m = x.shape

        partOne = 1.0 / ((2 * np.pi) ** (n / 2) * np.linalg.det(s) ** 0.5)
        partTwo = - 0.5 * np.dot(np.dot((x - mu).T, np.linalg.inv(s)), x - mu)
        p = partOne * np.exp(partTwo)
        return p

def getCircle(mu, s):
    points = 100

    center = mu
    radius = np.linalg.det(s)

    tt = np.array([np.linspace(0, 2 * np.pi, points)])  # 产生的点
    x = center[0, :] + np.cos(tt) * radius
    y = center[1, :] + np.sin(tt) * radius
    return x, y

if __name__ == '__main__':
    ggm = GGM2()
    l, params = ggm.globalEM()
    C = ggm.classifyData(params)

    colorPoint = ['r*', 'b+', 'g^', 'y.']
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.plot(ggm.D[0, :], ggm.D[1, :], 'b+')
    plt.subplot(1, 2, 2)
    plt.grid()
    for i in range(len(params['mu'])):
        for j in C[i]:
            plt.plot(ggm.D[0, j], ggm.D[1, j], colorPoint[i % len(colorPoint)])
        # cx, cy = getCircle(params['mu'][i], params['s'][i])
        # plt.plot(cx[0,:], cy[0,:], color[i % len(color)])

    plt.show()