from data import *
from smoStruct import *
from smoPlat import *

if __name__ == '__main__':
    dataCreated, dataLabel = GenerateData(500, "Guassion")
    # print(DataCreated)
    # print(DataLabel)
    smo = smoStruct(dataCreated, dataLabel, 10, 0.0001, ['rbf', 1.3])
    # print(b)
    # print(alpha)

    b, alpha = smo.exteriorLoop(20)
    testData, testLabel = GenerateData(20, "Guassion")

    newLabel = smo.funOtherX(dataCreated, kTup=['rbf', 1.3])
    # print(newLabel)

    for i in range(newLabel.shape[0]):
        if newLabel[i, 0] >= 1:
            plt.plot(dataCreated[i, 0], dataCreated[i, 1], 'r^')
            # plt.plot(testData[i, 0], testData[i, 1], 'r^')
        elif newLabel[i, 0] <= -1:
            plt.plot(dataCreated[i, 0], dataCreated[i, 1], 'g+')
            # plt.plot(testData[i, 0], testData[i, 1], 'g+')
        else:
            plt.plot(dataCreated[i, 0], dataCreated[i, 1], 'y')
            # plt.plot(testData[i, 0], testData[i, 1], 'y')

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.show()

# if __name__ == '__main__':
#     dataCreated, dataLabel = GenerateData(500, "Guassion")
#     b, alpha = smoP(dataCreated, dataLabel.T, 10000, 0.0001, 20)
#     testData, testLabel = GenerateData(20, "Guassion")
#
#     w = np.multiply(alpha, dataLabel).T * dataCreated
#
#     newLabel = dataCreated * w.T + b
#     print(newLabel)
#
#     for i in range(newLabel.shape[0]):
#         if newLabel[i, 0] >= 1:
#             plt.plot(dataCreated[i, 0], dataCreated[i, 1], 'r^')
#             # plt.plot(testData[i, 0], testData[i, 1], 'r^')
#         elif newLabel[i, 0] <= -1:
#             plt.plot(dataCreated[i, 0], dataCreated[i, 1], 'g+')
#             # plt.plot(testData[i, 0], testData[i, 1], 'g+')
#         else:
#             plt.plot(dataCreated[i, 0], dataCreated[i, 1], 'y')
#             # plt.plot(testData[i, 0], testData[i, 1], 'y')
#     plt.xlabel("X1")
#     plt.ylabel("X2")
#     plt.grid(True)
#     plt.show()