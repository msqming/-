import numpy as np
import matplotlib.pyplot as plt


def sigmoid(t):
    """ Sigmoid 函数 """
    return 1 / (1 + np.exp(-t))


# 生成测试数据 -10到10之间的500个数据
x = np.linspace(-10, 10, 500)
y = sigmoid(x)

# 绘制图像
plt.plot(x, y)
plt.show()
