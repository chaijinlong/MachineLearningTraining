import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from NaiveBayes import *

iris = datasets.load_iris()
irisData = iris['data']
irisLabel = iris['target']

# P_Y, P_X_Y = NaiveBayesLearning(irisData, irisLabel)
y = NaiveBayes(irisData, irisLabel, irisData[1, :])


c= 1