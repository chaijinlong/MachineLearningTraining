import numpy as np

def NaiveBayesLearning(data, label, type=''):
    if type == '' or type == 'discrete':
        N = len(label) # 训练数据的数量
        labelSet = set(label) # 标记的元素
        featureCount = data.shape[1] # 特征的数量
        featureSet = [] # 每个特征的可能取值
        P_X_Y = {} # 条件概率
        P_Y = {} # 先验概率

        for eachLabel in labelSet:
            P_Y[eachLabel] = list(label).count(eachLabel) / N

        for j in range(featureCount):
            featureSet.append(list(set(data[:, j]))) # 第j个特征的可能取值

        for eachLabel in labelSet:
            eachLabelData = np.array([list(data)[row] for row in range(data.shape[0]) if label[row] == eachLabel])
            for j in range(data.shape[1]):
                for each in featureSet[j]:
                    P_X_Y[(eachLabel, j, each)] = len(np.where(eachLabelData[:, j] == each)[0]) / P_Y[eachLabel] / N

        return P_Y, P_X_Y

def NaiveBayes(data, label, x='', type=''):
    if type == '' or type == 'discrete':
        P_Y, P_X_Y = NaiveBayesLearning(data, label, type)
        y = {}
        for eachP_Ykeys in P_Y.keys():
            temp = 1
            for j in range(data.shape[1]):
                temp = temp * P_X_Y[(eachP_Ykeys, j, x[j])]
            y[eachP_Ykeys] = temp * eachP_Ykeys
        return list(y.keys())[list(y.values()).index(max(y.values()))]
