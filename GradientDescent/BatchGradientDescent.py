import numpy as np

def punishFun(fun, learningRate=None):
    '''
    :feature: 计算惩罚函数的值

    :param fun: 待拟合的多项式对象
    :param learningRate: 学习速率
    :return: fun.deviation: 惩罚函数的值
    '''
    if learningRate == None:
        learningRate = 0.5 / fun.xVector.size

    fun.deviation = learningRate * np.sum((fun.getY(fun.xVector) - fun.yVector) ** 2)
    return fun.deviation

def partPunishFun(fun, learningRate=None):
    '''
    :feature: 计算惩罚函数对不同$\theta$的偏导数，即梯度

    :param fun: 待拟合的多项式对象
    :param learningRate: 学习速率
    :return: partPun: 不同$\theta$的梯度组成的向量
    '''
    if learningRate == None:
        learningRate = 0.5 / fun.xVector.size

    partPun = np.zeros(fun.wVector.size)
    for index in range(0, fun.wVector.size):
        partPun[index] = 2 * learningRate * np.sum((fun.getY(fun.xVector) - fun.yVector) * fun.partPloyFunbyW(index))
    return partPun

def updateTheta(fun, punLimit, iterLimit, learningRate=None):
    '''
    :feature: 更新多项式的次数$\theta$

    :param fun: 待拟合的多项式对象
    :param punLimit: 惩罚限制
    :param iterLimit: 迭代限制
    :param learningRate: 学习速率
    :return: fun.deviation: 惩罚函数的值
    '''
    iterNum = 0
    while punishFun(fun, learningRate) > punLimit and iterNum < iterLimit:
        iterNum += 1
        partPun = partPunishFun(fun, learningRate)
        fun.wVector -= partPun

    return fun.deviation