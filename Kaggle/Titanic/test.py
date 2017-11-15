import pandas as pd 
import matplotlib.pyplot as plt 

def loadCsv(fileName=None):
	if fileName == None:
		print('Error: file is not exist!')
	else:
		dataTrain = pd.read_csv(fileName)
		return dataTrain


# def preprocessing(dataTrain):

if __name__ == '__main__':
	dataTrain = loadCsv('train.csv')
	print(dataTrain)
	
	# df = pd.DataFrame({'Fare': dataTrain.Fare, 'Survived': dataTrain.Survived})
	# df = df.sort_values('Fare')
	# plt.plot(df.Fare, df.Survived)
	# plt.show()
