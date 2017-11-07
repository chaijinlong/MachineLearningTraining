def cart(data, attr):
	import myClass
	node = myClass.Node()

### 停止条件
	if dataIsSameInClassify(data):
	# 如果所有的训练集的分类都相同时，结束递归
		node.changeClassify(data[0]['classify'])
		return(node)

	if dataIsSameInAttr(data, attr):
	# 如果所有训练集上当前属性集的元素相同，结束递归
		node.changeClassify(mostClassifyInData(data))

	if not attr:
	# 如果属性集为空
		node.changeClassify(mostClassifyInData(data))
###

	giniData = getSplitAttr(data, attr)
	node.changeAttribute(str(giniData[0]) + ': ' + str(giniData[1]))
	dataTrue, dataFalse = splitData22(data, giniData[0], giniData[1])
	attrCopy = attr.copy()
	attrCopy.pop(giniData[0])
	node.changeLeftChild(cart(dataTrue, attrCopy))
	node.changeRightChild(cart(dataFalse, attrCopy))
	return(node)


def mostClassifyInData(data):
	mostClassify = []
	import collections
	countData = dict(collections.Counter([each_data['classify'] for each_data in data]))
	mostClassifyNum = max(list(countData.values()))
	for eachCountData in countData:
		if countData[eachCountData] == mostClassifyNum:
			mostClassify.append(eachCountData)
	return(mostClassify)


def dataIsSameInAttr(data, attr):
	for eachData in data:
		for eachAttrKey in attr.keys():
			if eachData['attrValue'][eachAttrKey] != data[0]['attrValue'][eachAttrKey]:
				return(False)
	return(True)


def dataIsSameInClassify(data):
	for eachData in data:
		if eachData['classify'] != data[0]['classify']:
			return(False)
	return(True)


def getSplitAttr(data, attr):
	giniAttr = {}
	giniMin = 1
	for eachKey in attr.keys():
		bestAttribute, gini = giniSplit(data, eachKey, attr[eachKey])
		# 对于属性eachKey计算其最小的gini系数及其对应的划分属性的元素
		if gini < giniMin:
			giniMin = gini
			giniAttr = [eachKey, bestAttribute, giniMin]
			# 纪录最小的gini系数的属性及其对应的划分属性的元素
	return(giniAttr)


def giniSplit(data, attributeKey, attributeValue):
# 输入：data 训练集；attributeKey 属性A的名字；attributeValue 属性A的所有元素的列表
# 输出：bestAttribute, giniMin 最小的gini系数及其对应的划分属性的元素
	giniMin = 1
	for eachAttributeValue in attributeValue:
	# 对于属性A中的每一个元素
		dataTrue, dataFalse = splitData22(data, attributeKey, eachAttributeValue)
		giniS = (len(dataTrue) * gini(dataTrue) + len(dataFalse) * gini(dataFalse)) / len(data)
		if giniS < giniMin:
			giniMin = giniS
			bestAttribute = eachAttributeValue
	return(bestAttribute, giniMin)


def splitData22(data, attributeKey, value):
# 输入：data 训练集；attrKey 属性的名字，value 属性的某一个元素的值
# 输出：(dataTrue, dataFalse) attrKey属性的值和value相等的列表，attrKey属性的值和value不相等的列表
	dataTrue = []
	dataFalse = []
	for eachData in data:
		if eachData['attrValue'][attributeKey] == value:
			# 如果训练集(data)中的属性值(attrValue)的attrKey属性的值和value相等
			dataTrue.append(eachData)
		else:
			dataFalse.append(eachData)
	return(dataTrue, dataFalse)


def gini(data):
# 输入：data 训练集
# 输出：gini gini值
	gini = 1
	temp = {} # 创建一个dict，用于储存data中不同类别子集的数量
	for eachData in data:
		if temp.get(eachData['classify']) is None:
			# 如果temp中没有eachData['classify']这个key时
			temp.update({eachData['classify']: 1})
		else:
			temp[eachData['classify']] += 1
	for eachValue in temp.values():
		gini -= (eachValue / len(data)) ** 2
	return(gini)
