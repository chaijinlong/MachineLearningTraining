import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import preprocessing, metrics
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

def loadCsv(fileName=None):
    if fileName == None:
        print('Error: file is not exist!')
    else:
        dataTrain = pd.read_csv(fileName)
        return dataTrain

def data_preprocess():
    dataTrain = loadCsv('train.csv')
    dataTest = loadCsv('test.csv')

    dataTrain['Age'] = dataTrain['Age'].fillna(dataTrain['Age'].median()) # 插入平均值
    dataTrain['Age'] = preprocessing.scale(dataTrain['Age'])

    dataTrain.loc[dataTrain['Sex'] == 'male', 'Sex'] = 0
    dataTrain.loc[dataTrain['Sex'] == 'female', 'Sex'] = 1
    dataTrain['Sex'] = preprocessing.scale(dataTrain['Sex'])

    dataTrain.loc[dataTrain['Embarked'].isnull(), 'Embarked'] = 0
    dataTrain.loc[dataTrain['Embarked'] == 'S', 'Embarked'] = 0
    dataTrain.loc[dataTrain['Embarked'] == 'Q', 'Embarked'] = 1
    dataTrain.loc[dataTrain['Embarked'] == 'C', 'Embarked'] = 2
    dataTrain['Embarked'] = preprocessing.scale(dataTrain['Embarked'])

    dataTrain['Fare'] = preprocessing.scale(dataTrain['Fare'])
    dataTrain['SibSp'] = preprocessing.scale(dataTrain['SibSp'])
    dataTrain['Parch'] = preprocessing.scale(dataTrain['Parch'])

    dataTest['Age'] = dataTest['Age'].fillna(dataTest['Age'].median()) # 插入平均值
    dataTest['Age'] = preprocessing.scale(dataTest['Age'])

    dataTest.loc[dataTest['Sex'] == 'male', 'Sex'] = 0
    dataTest.loc[dataTest['Sex'] == 'female', 'Sex'] = 1
    dataTest['Sex'] = preprocessing.scale(dataTest['Sex'])

    dataTest.loc[dataTest['Embarked'].isnull(), 'Embarked'] = 0
    dataTest.loc[dataTest['Embarked'] == 'S', 'Embarked'] = 0
    dataTest.loc[dataTest['Embarked'] == 'Q', 'Embarked'] = 1
    dataTest.loc[dataTest['Embarked'] == 'C', 'Embarked'] = 2
    dataTest['Embarked'] = preprocessing.scale(dataTest['Embarked'])

    dataTest['Fare'] = dataTest['Fare'].fillna(dataTest['Fare'].median())
    dataTest['Fare'] = preprocessing.scale(dataTest['Fare'])

    dataTest['SibSp'] = preprocessing.scale(dataTest['SibSp'])
    dataTest['Parch'] = preprocessing.scale(dataTest['Parch'])

    return dataTrain, dataTest

def main_test(n_estimators, max_features):
    dataTrain, dataTest = data_preprocess()

    predictors = ['Pclass', 'Sex', 'Fare', 'Embarked', 'Age', 'SibSp', 'Parch']

    clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, oob_score=True)

    scores = cross_validation.cross_val_score(clf, dataTrain[predictors], dataTrain['Survived'], cv=5)
    cross_validation.cross_val_score(clf, dataTrain[predictors], dataTrain['Survived'], cv=5)

    print(str(n_estimators) + ', ' + str(max_features) + ': ' + str(scores) + ', mean: ' + str(np.mean(scores)))

    return np.mean(scores)

    # clf.fit(dataTrain[predictors], dataTrain['Survived'])
    # predict = clf.predict(dataTest[predictors])

    # result = pd.DataFrame({'PassengerId':dataTest['PassengerId'].as_matrix(), 'Survived':predict.astype(np.int32)})
    # result.to_csv("/Users/chai/Github/MachineLearningTraining/Kaggle/Titanic/RF.csv", index=False)

def main(n_estimators, max_features):
    dataTrain, dataTest = data_preprocess()

    predictors = ['Pclass', 'Sex', 'Fare', 'Embarked', 'Age', 'SibSp', 'Parch']

    clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, oob_score=True)

    clf.fit(dataTrain[predictors], dataTrain['Survived'])
    predict = clf.predict(dataTest[predictors])

    result = pd.DataFrame({'PassengerId':dataTest['PassengerId'].as_matrix(), 'Survived':predict.astype(np.int32)})
    result.to_csv("/Users/chai/Github/MachineLearningTraining/Kaggle/Titanic/RF.csv", index=False)

if __name__ == '__main__':
    main_test(n_estimators=110, max_features="log2")
    main(n_estimators=110, max_features="log2")

    # scores = 0
    # para = []

    # for n_estimators in range(40):
    #     for max_features in ["auto", "log2", None]:
    #         temp = main(n_estimators=10 * n_estimators + 100, max_features=max_features)
    #         if temp > scores:
    #             scores = temp
    #             para = [10 * n_estimators + 100, max_features]

    # print(str(scores) + ': ' + str(para))
