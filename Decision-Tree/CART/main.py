import myTools

import myClass

import cart

import printTree


attrName = ['体温','表面覆盖','胎生','产蛋','能飞','水生','有腿','冬眠']
data = myTools.getDataFromFile('animals.txt', attrName)
attr = myTools.getAllAttrValue(data, attrName)

root = cart.cart(data, attr)

printTree.printAll(root, [0, 0], 100, 5)