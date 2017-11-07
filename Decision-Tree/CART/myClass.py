class Node:
	def __init__(self, deepNum = -1, attribute = None, classify = None, data = None, leftChild = None, rightChild = None):
		self.deepNum = deepNum
		self.classify = classify
		self.attribute = attribute
		self.leftChild = leftChild
		self.rightChild = rightChild

		if data is None:
			self.data = []
		else:
			self.data = data


	def changeDeepNum(self, deepNum = -1):
		if deepNum == -1:
			pass
		else:
			self.deepNum = deepNum

	def changeData(self, data = None):
		if data is None:
			pass
		else:
			self.data = data

	def changeAttribute(self, attribute = None):
		if attribute is None:
			pass
		else:
			self.attribute = attribute

	def changeClassify(self, classify = None):
		if classify is None:
			pass
		else:
			self.classify = classify

	def changeLeftChild(self, leftChild = None):
		if leftChild is None:
			pass
		else:
			self.leftChild = leftChild

	def changeRightChild(self, rightChild = None):
		if rightChild is None:
			pass
		else:
			self.rightChild = rightChild

	def printAll(self):
		print('attribute: ' + str(self.attribute) + '; classify: ' + str(self.classify) + '; leftChild: ' + str(self.leftChild) + '; rightChild: ' + str(self.rightChild))




















