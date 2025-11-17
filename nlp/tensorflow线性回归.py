# function print study
# create by gather
# create time 2025/8/30
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 20) + np.random.randn(20)
y = np.linspace(0, 10, 20) + + np.random.randn(20)
plt.scatter(x, y)
plt.show()

# 定义W 和 B
W = tf.Variable(np.random.randn() * 0.02)  # 随机初始化
B = tf.Variable(0.)


# 定义线性模型
def linear_regression(temp_x):
    return W * temp_x + B


# 定义损失函数
def mean_square_loss(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))


# 定义优化器
optimizer = tf.keras.optimizers.SGD()  # 批量梯度下降


# 定义优化过程
def run_optimizers():
    with tf.GradientTape() as g:
        pred = linear_regression(x)  # 模型
        loss = mean_square_loss(pred, y)  # 损失函数
    # 计算loss 函数 对[W, B]的梯度
    gradients = g.gradient(loss, [W, B])  # 返回梯度
    # 根据梯度，更新W，B
    optimizer.apply_gradients(zip(gradients, [W, B]))


# 训练
for step in range(5000):
    run_optimizers()  # 训练一次
    if 0 == step % 100:
        pred = linear_regression(x)
        loss = mean_square_loss(pred, y)
        print(f"step:{step} loss:{loss} W:{W.numpy()} B:{B.numpy()}")
