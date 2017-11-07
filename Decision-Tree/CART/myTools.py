def getDataFromFile(filename, attrName):
	with open(filename, 'r') as openFile:
		data = []
		num = 0
		for eachLine in openFile:
			if eachLine != '\n':
				eachData = {'serialNum': None, 'classify': None, 'attrValue': None}
				eachData['serialNum'] = num
				eachData['attrValue']= dict(zip(attrName, eachLine.strip().split(',', 10)[:-1]))
				eachData['classify'] = eachLine.strip().split(',', 10)[-1]
				data.append(eachData)
				num += 1
			else:
				pass
	return(data)


def getAllAttrValue(data, attrName):
	for eachData in data:
		if len(attrName) != len(list(eachData['attrValue'].values())):
			print('(By Chai) Error In myTools.py: Line 26')
			return
	attr = {}
	for eachAttrName in attrName:
		temp = []
		for eachData in data:
			temp.append(eachData['attrValue'][eachAttrName])
		attr.update({eachAttrName: list(set(temp))})
	return(attr)


