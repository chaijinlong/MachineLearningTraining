print(__doc__)

from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()

plt.figure(1, figsize=(3,3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()