import numpy as np


def creatDate(nx):
    x = np.random.rand(nx, 1) * 10 + np.array([[1, 1]])
    for i in range((int)(np.random.rand(1)[0] * 10) + 40):
        x = np.hstack((x, np.random.rand(nx, 1) * 5 + np.array([[1], [1]])))

    for i in range((int)(np.random.rand(1)[0] * 10) + 40):
        x = np.hstack((x, np.random.rand(nx, 1) * 5 + np.array([[50], [50]])))

    for i in range((int)(np.random.rand(1)[0] * 10) + 40):
        x = np.hstack((x, np.random.rand(nx, 1) * 5 + np.array([[50], [1]])))
    #
    # for i in range((int)(np.random.rand(1)[0] * 10) + 40):
    #     x = np.hstack((x, np.random.rand(nx, 1) * 5 + np.array([[1], [50]])))
    # print(x.shape)
    return x