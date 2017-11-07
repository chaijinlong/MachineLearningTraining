import numpy as np
from data import *
import matplotlib.pyplot as plt

def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif L > aj:
        aj = L
    return aj

def smoSimple(dataMatrix, classLabels, C, toler, maxIter):
    b = 0
    m, n = dataMatrix.shape
    alpha = np.mat(np.zeros(m)).T
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alpha, classLabels).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(classLabels[i])
            if ((classLabels[i] * Ei < -toler) and (alpha[i] < C) or (classLabels[i] * Ei > toler) and (alpha[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(np.multiply(alpha, classLabels).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(classLabels[j])
                alphaIOld = alpha[i].copy()
                alphaJOld = alpha[j].copy()
                if (classLabels[i] != classLabels[j]):
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H:
                    print("eta>=0")
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T \
                      - dataMatrix[j, :] * dataMatrix[j, :].T
                alpha[j] -= classLabels[j] * (Ei - Ej) / eta
                alpha[j] = clipAlpha(alpha[j], H, L)
                if (abs(alpha[j] - alphaJOld) < 0.00001):
                    print("j not moving enoutgh")
                alpha[i] += classLabels[j] * classLabels[i] * (alphaJOld - alpha[j])
                b1 = b - Ei - classLabels[i] * (alpha[i] - alphaIOld) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     classLabels[j] * (alpha[j] - alphaJOld) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - classLabels[i] * (alpha[i] - alphaIOld) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                    classLabels[j] * (alpha[j] - alphaJOld) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alpha[i] < C):
                    b = b1
                elif (0 < alpha[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alpha

if __name__ == '__main__':
    dataCreated, dataLabel = GenerateData(100, "Guassion")
    # print(DataCreated)
    # print(DataLabel)
    b, alpha = smoSimple(dataCreated, dataLabel, 1000000, 0.001, 100)
    # print(b)
    # print(alpha)

    testData, testLabel = GenerateData(40, "Guassion")

    w1 = np.multiply(alpha, dataLabel).T * dataCreated[:, 0]
    w2 = np.multiply(alpha, dataLabel).T * dataCreated[:, 1]
    w = np.vstack((w1, w2))

    newLabel = np.array(testData * w) + b
    print(newLabel)

    for i in range(newLabel.shape[0]):
        if newLabel[i] >= 1:
            plt.plot(testData[i, 0], testData[i, 1], 'r^')
        if newLabel[i] <= -1:
            plt.plot(testData[i, 0], testData[i, 1], 'gs')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show()