def id3(data, attr):
	import myClass

	node = myClass.Node()
	node.changeData(data)
	deepNum = 4 - len(attr)
	node.changeDeepNum(deepNum)

	temp = []
	for eachData in data:
		temp.append(eachData['classify'])

	if len(set(temp)) == 1:

		node.changeClassify(temp[0])
		return(node)

	if (not list(attr.keys())) | dataIsSameInAttr(data, attr):

		node.changeClassify(mostClassifyInData(data))
		return(node)

	bestAttr = findMaxGain(data, attr)
	node.changeAttribute(list(bestAttr.keys())[0])

	for eachValueInBestAttr in bestAttr[list(bestAttr.keys())[0]]:
		dataV = getDataV(data, bestAttr, eachValueInBestAttr)
		if not dataV:
			nextNode = myClass.Node()
			nextNode.changeClassify(mostClassifyInData(data))
			nextNode.changeDeepNum(deepNum + 1)
			node.addNextNode({eachValueInBestAttr: nextNode})
		else:
			attrCopy = attr.copy()
			attrCopy.pop(list(bestAttr.keys())[0])
			node.addNextNode({eachValueInBestAttr: id3(dataV, attrCopy)})
	return(node)

def dataIsSameInAttr(data, attr):
	dataAttrValue = []
	for eachData in data:
		temp = {}
		for eachAttrKey in attr.keys():
			temp.update({eachAttrKey: eachData['attrValue'][eachAttrKey]})
		dataAttrValue.append(temp)
	for eachdataAttrValue in dataAttrValue:
		if eachdataAttrValue == dataAttrValue[0]:
			pass
		else:
			return(False)
	return(True)

def mostClassifyInData(data):
	mostClassify = []
	import collections
	countData = dict(collections.Counter([each_data['classify'] for each_data in data]))
	mostClassifyNum = max(list(countData.values()))
	for eachCountData in countData:
		if countData[eachCountData] == mostClassifyNum:
			mostClassify.append(eachCountData)
	return(mostClassify)

def getDataV(data, bestAttr, attrValue):
	dataV = []
	for eachData in data:
		if eachData['attrValue'][list(bestAttr.keys())[0]] == attrValue:
			dataV.append(eachData)
	return(dataV)

def findMaxGain(data, attr):
	temp = gain(data, attr)
	tempValues = list(temp.values())
	for tempKey in temp.keys():
		if temp[tempKey] == max(tempValues):
			return({tempKey: attr[tempKey]})

def gain(data, attr):
	temp = {}
	for eachAttrKey in attr.keys():
		sum = entropy(data)
		for eachAttrValue in attr[eachAttrKey]:
			dataV = getDataV(data, {eachAttrKey :attr[eachAttrKey]}, eachAttrValue)
			sum -= len(dataV) / len(data) * entropy(dataV)
		temp.update({eachAttrKey: sum})
	return(temp)

def entropy(data):
	import math
	import collections
	countData = dict(collections.Counter([each_data['classify'] for each_data in data]))
	ent = 0
	for countDataKey in countData.keys():
		ent -= countData[countDataKey] / len(data) * math.log2(countData[countDataKey] / len(data))
	return(ent)