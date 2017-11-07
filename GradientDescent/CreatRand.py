import numpy as np

def creatGaussRand(num=1, mean=0, var=1):
    return np.random.normal(mean, var, num)