import PolynomicalFun
import BatchGradientDescent
import StochasticGradientDescent
import numpy as np
from CreatRand import *
import matplotlib.pyplot as plt
from time import clock

numOfTrainAssemble = 1000

xVector = np.array(list(range(0, numOfTrainAssemble)), float) / numOfTrainAssemble
yVector = np.sin(xVector * np.pi * 2.0) + creatGaussRand(numOfTrainAssemble, 0, 0.1)

theta = np.zeros(5)
fun = PolynomicalFun.PolynomicalFun(xVector, yVector, theta)

BGDstart = clock()
print(BatchGradientDescent.updateTheta(fun, 0.01, 10000))
print(str((clock() - BGDstart) / 1000000) + 's')

theta = np.zeros(5)
fun = PolynomicalFun.PolynomicalFun(xVector, yVector, theta)

SGDstart = clock()
print(StochasticGradientDescent.updateTheta(fun, 0.01, 1))
print(str((clock() - SGDstart) / 1000000) + 's')

#
# theta = StochasticGradientDescent.updateTheta(fun, 0.01, 100)
#
# pVector = fun.getY(xVector)
#
# x0Vector = np.array(list(range(0, 10000)), float) / 10000
# y0Vector = np.sin(x0Vector * np.pi * 2.0)
#
# plt.figure(1)
# plt.grid()
# plt.plot(x0Vector, y0Vector, 'r', linewidth=1) # 目标函数
# #plt.plot(xVector, yVector, 'yo', linewidth=2) # 观测值
# plt.plot(xVector, pVector, 'b--', linewidth=1) # 拟合的函数
# plt.show()