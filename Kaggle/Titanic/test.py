import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn import preprocessing, metrics
import numpy as np 

def loadCsv(fileName=None):
	if fileName == None:
		print('Error: file is not exist!')
	else:
		dataTrain = pd.read_csv(fileName)
		return dataTrain

if __name__ == '__main__':
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

	predictors = ['Pclass', 'Sex', 'Fare', 'Embarked', 'Age', 'SibSp', 'Parch']

	clf = SVC()
	clf.fit(dataTrain[predictors], dataTrain['Survived'])

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

	predict = clf.predict(dataTest[predictors])

	result = pd.DataFrame({'PassengerId':dataTest['PassengerId'].as_matrix(), 'Survived':predict.astype(np.int32)})
	result.to_csv("/Users/chai/Github/MachineLearningTraining/Kaggle/Titanic/SVM_predictions.csv", index=False)

	# print('Accuracy for SVM is ' + metrics.accuracy_score(predict, dataTest['Survived']))