import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.multiclass import OneVsOneClassifier

# 鸢尾花数据集
iris = datasets.load_iris()
# 去数据集中的前两列属性值
X = iris.data[:, :2]
y = iris.target

# 分割测试数据集和训练数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

log_reg = LogisticRegression()
# 利用训练数据集进行训练
ovo = OneVsOneClassifier(log_reg)
ovo.fit(X_train, y_train)
print('数据集前两个属性的准确率',ovo.score(X_test, y_test))  # 0.84


def plot_decision_boundary(model, axis):
    """ 画图函数 """
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


plot_decision_boundary(ovo, axis=[4, 8.5, 1.5, 4.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])
plt.show()

X_all = iris.data
y_all = iris.target

# 分割测试数据集和训练数据集
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state=666)

log_reg = LogisticRegression()
# 利用训练数据集进行训练
ovo = OneVsOneClassifier(log_reg)
ovo.fit(X_train, y_train)
print('数据集前全部属性的准确率',ovo.score(X_test, y_test))  # 1.0