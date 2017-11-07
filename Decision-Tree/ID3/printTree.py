import matplotlib.pyplot as plt

def printAll(node, startPoint = None, width = 0):
	plt.axis([startPoint[0] - 3, width + 3, -23, 8])
	printTree(node, startPoint, width)

	plt.show()

def printTree(node, startPoint = None, width = 0):
	height = 5

	if startPoint is None:
		return
	else:
		plotBox(node, startPoint, width, height)
		if (node.childNode is None):
			pass
		else:
			childNum = len(node.childNode)
			childStartPoint = [startPoint[0], startPoint[1] - 10]
			for childKey in node.childNode.keys():
				printTree(node.childNode[childKey], childStartPoint, width / childNum / 2)
				tump = printBranch([startPoint[0] + width / 2, startPoint[1]], [childStartPoint[0], childStartPoint[1] + height])
				plt.text(tump[0], tump[1], childKey)
				childStartPoint = [childStartPoint[0] + width / childNum, childStartPoint[1]]


def plotBox(node, point = None, width = 0, height = 0):
	if point is None:
		return(-1)
	x = point[0]
	y = point[1]

	plt.plot([x, x + width], [y, y], 'b')
	plt.plot([x, x], [y, y + height], 'b')
	plt.plot([x, x + width], [y + height, y + height], 'b')
	plt.plot([x + width, x + width], [y + height, y], 'b')

	printText(node, [x + width / 3, y + height / 3])
	return(1)

def printBranch(aPoint = None, bPoint = None):
	if (aPoint is None) | (bPoint is None):
		return(-1)

	plt.plot([aPoint[0], bPoint[0]], [aPoint[1], bPoint[1]])
	return([(aPoint[0] + bPoint[0]) / 2, (aPoint[1] + bPoint[1]) / 2])


def printText(node, point = None):
	if point is None:
		return(-1)

	if (node.attribute is None):
		plt.text(point[0], point[1], str(node.classify))
	elif (node.classify is None):
		plt.text(point[0], point[1], str(node.attribute))
	else:
		plt.text(point[0], point[1], 'Error!!!')
		return(1)