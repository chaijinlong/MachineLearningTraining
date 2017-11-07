import myTools

import myClass

import id3

import printTree

attrName = ['体温','表面覆盖','胎生','产蛋','能飞','水生','有腿','冬眠']
data = myTools.getDataFromFile('animals.txt', attrName)

'''
attrName = ['outlook', 'temperature', 'humidity', 'windy']
data = myTools.getDataFromFile('weather.txt', attrName)
'''

attr = myTools.getAllAttrValue(data, attrName)
root = id3.id3(data, attr)

root.printAll()
for child in root.childNode.values():
	child.printAll()
	for childchild in child.childNode.values():
		childchild.printAll()

printTree.printAll(root, [0, 0], 100)