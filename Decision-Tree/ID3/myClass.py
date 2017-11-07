class Node:
	def __init__(self, deepNum = -1, attribute = None, classify = None, data = None, childNode = None):
		self.deepNum = deepNum
		self.attribute = attribute
		self.classify = classify
		if data is None:
			self.data = []
		else:
			self.data = data

		if childNode is None:
			self.childNode = {}
		else:
			self.childNode = childNode

	def changeDeepNum(self, deepNum = -1):
		self.deepNum = deepNum

	def changeData(self, data):
		self.data = data

	def changeAttribute(self, attribute = None):
		self.attribute = attribute

	def changeClassify(self, classify = None):
		self.classify = classify

	def changeNextNode(self, nextNode = None):
		if nextNode is None:
			self.childNode = {}
		else:
			self.childNode = nextNode

	def addNextNode(self, nextNode = None):
		if nextNode is None:
			self.childNode = {}
		else:
			self.childNode.update(nextNode)

	def printAll(self):
		print(str(self.deepNum) + ': attribute = ' + str(self.attribute) + '; classify = ' + str(self.classify) + '; childNode = ' + str(self.childNode))