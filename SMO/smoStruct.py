import numpy as np

class smoStruct:
    def __init__(self, xMatrix, label, C, toler, kTup):
        self.xMatrix = xMatrix # [m,n]
        self.label = label # [m, 1]
        self.C = C # float
        self.toler = toler # float
        self.m = xMatrix.shape[0] # inteter
        self.n = xMatrix.shape[1] # inteter

        self.alpha = np.mat(np.zeros((self.m, 1))) # [m,1]
        self.b = 0 # float
        self.eCache = np.mat(np.zeros((self.m, 2))) # [m,2]

        self.K = np.mat(np.zeros((self.m, self.m))) # [m,m]
        for i in range(self.m):
            for j in range(self.m):
                a = self.kernelTrans(self.xMatrix[i, :], self.xMatrix[j, :], kTup)
                self.K[i, j] = self.kernelTrans(self.xMatrix[i, :], self.xMatrix[j, :], kTup)

    def funX(self, index=None):
        fX = 0
        if index == None:
            fX = (np.multiply(self.alpha, self.label).T * self.K.T).T + self.b
        elif isinstance(index, int) and (0 <= index <= self.m):
            fX = (np.multiply(self.alpha, self.label).T * self.K[index, :].T + self.b)[0, 0]
        return fX

    def funOtherX(self, otherMatrix='', index=None, kTup=''):
        if index == None:
            fX = np.zeros((otherMatrix.shape[0], 1))
            for index in range(otherMatrix.shape[0]):
                for i in range(self.m):
                    fX[index, 0] += self.alpha[i, 0] * self.label[i, 0] * self.kernelTrans(otherMatrix[index, :], self.xMatrix[i, :], kTup)
                fX[index, 0] += self.b
            return fX
        else:
            fX = 0
            for i in range(self.m):
                fX += self.alpha[i, 0] * self.label[i, 0] * self.kernelTrans(otherMatrix[index, :], self.xMatrix[i, :], kTup)
            fX += self.b
            return fX[0, 0]

    def calE(self, index):
        return self.funX(index) - self.label[index, 0] # inteter

    def updateE(self, index):
        self.eCache[index] = [1, self.calE(index)]

    def selectJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0.0
        self.eCache[i] = [1, Ei]
        validECacheList = np.nonzero(self.eCache[:, 0].A)[0]
        if len(validECacheList) > 1:
            for k in validECacheList:
                if k == i:
                    continue
                Ek = self.calE(k)
                deltaE = np.abs(Ei - Ek)
                if deltaE > maxDeltaE:
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = i
            while j == i:
                j = int(np.random.uniform(0, self.m))
                Ej = self.calE(j)
            return j, Ej

    def interiorLoop(self, i):
    # 内部循环
        Ei = self.calE(i)
        # 更新误差
        if ((self.label[i, 0] * Ei < -self.toler) and (self.alpha[i, 0] < self.C))\
            or ((self.label[i, 0] * Ei > self.toler) and (self.alpha[i, 0] > 0)):
        # 违反KKT条件
            j, Ej = self.selectJ(i, Ei)
            # 根据第一个参数选择合适的第二个参数
            alphaIOld = self.alpha[i, 0].copy()
            alphaJOld = self.alpha[j, 0].copy()
            if (self.label[i, 0] != self.label[j, 0]):
                L = max(0, self.alpha[j, 0] - self.alpha[i, 0])
                H = min(self.C, self.C + self.alpha[j, 0] - self.alpha[i, 0])
            else:
                L = max(0, self.alpha[j, 0] + self.alpha[i, 0] - self.C)
                H = min(self.C, self.alpha[j, 0] + self.alpha[i, 0])
            # 根据约束条件，alpha必须在正方形内部的直线上
            if H == L:
                return 0
            eta = (2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j])
            if eta >= 0:
            # 判断eta，如果eta大于0，则函数是凸函数，没有极大值；如果eta等于0，则函数是线性的
                return 0
            self.alpha[j, 0] -= self.label[j, 0] * (Ei - Ej) / eta
            # 修改alpha_j的值
            if self.alpha[j, 0] > H:
                self.alpha[j, 0] = H
            elif self.alpha[j, 0] < L:
                self.alpha[j, 0] = L

            self.updateE(j)
            if abs(self.alpha[j, 0] - alphaJOld) < 0.00001:
                return 0

            self.alpha[i, 0] += self.label[j, 0] * self.label[i, 0] * (alphaJOld - self.alpha[j, 0])
            # 修改alpha_i的值
            self.updateE(i)

            b1 = self.b - Ei - self.label[i, 0] * (self.alpha[i, 0] - alphaIOld) * self.K[i, i] -\
                 self.label[j, 0] * (self.alpha[j, 0] - alphaJOld) * self.K[i, j]

            b2 = self.b - Ej - self.label[i, 0] * (self.alpha[i, 0] - alphaIOld) * self.K[i, j] -\
                 self.label[j, 0] * (self.alpha[j, 0] - alphaJOld) * self.K[j, j]

            if 0 < self.alpha[i, 0] < self.C:
                self.b = b1
            elif 0 < self.alpha[j, 0] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def exteriorLoop(self, maxIter):
    # 外部循环
        iter = 0
        # 迭代次数
        entireSet = True
        # 是否遍历了整个数据集
        alphaPairChanged = 0
        # 修改的alpha对的数量
        # self.updateE()
        # # 初始化计算误差
        while (iter < maxIter) and ((alphaPairChanged > 0) or entireSet):
        # 循环条件：迭代次数小于最大迭代次数，并且数据集没有alpha对改变或者整个数据集被循环了一次
            alphaPairChanged = 0
            if entireSet:
            # 循环整个数据集
                for i in range(self.m):
                # i可以是数据集中的每一个
                    alphaPairChanged += self.interiorLoop(i)
                print('FullSet, iter: %d i: %d, Pairs changed: %d' % (iter, i, alphaPairChanged))
                iter += 1
            else:
            # 修改不在边界上的点
                nonBoundPoint = np.nonzero((self.alpha.A > 0) * (self.alpha.A < self.C))[0]
                # 0<alpha<C的点，即不在边界上
                for i in nonBoundPoint:
                    alphaPairChanged += self.interiorLoop(i)
                    print('Non-bound, iter: %d i: %d, Pairs changed: %d' % (iter, i, alphaPairChanged))
                iter += 1
            if entireSet:
                entireSet = False
            elif (alphaPairChanged == 0):
                entireSet = True
            print('Iteration number: ' + str(iter))
        return self.b, self.alpha

    def kernelTrans(self, Xi, Xj, kTup):
        K = 0
        if kTup[0] == 'lin':
            K = Xi * Xj.T
        elif kTup[0] == 'rbf':
            deltaRow = Xi - Xj
            K = deltaRow * deltaRow.T
            K = np.exp(K / (-1 * kTup[1] ** 2))
        return K
