import numpy as np

class PolynomicalFun:
    xVector = np.array([0])
    yVector = np.array([0])
    wVector = np.array([0])
    deviation = np.inf

    def __init__(self, xVector, yVector, wVector):
        self.xVector = xVector
        self.yVector = yVector
        self.wVector = wVector

    def getY(self, x=""):
        if not isinstance(self.wVector, np.ndarray):
            return None

        if not isinstance(x, (np.ndarray, int, float)):
            return None

        if isinstance(x, np.ndarray):
            y = np.zeros(len(x))
            for index in range(0, self.wVector.size):
                y += self.wVector[index] * x ** index
            return y

        if isinstance(x, (int, float)):
            y = 0
            for index in range(0, self.wVector.size):
                y += self.wVector[index] * x ** index
            return y

    def partPloyFunbyW(self, index, num=None):
        if num == None:
            if isinstance(index, int):
                return self.xVector ** index
            if isinstance(index, (list, np.ndarray)):
                temp = np.zeros(len(index))
                for i in index:
                    temp[i] = self.xVector ** index[i]
                return temp
        elif isinstance(num, int):
            return self.xVector[num] ** index