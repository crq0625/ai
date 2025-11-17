# function print study
# create by gather
# create time 2025/8/30

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
"""
找到预测函数：1/(1+e^-z)
找到损失函数：-(y_true)log(y_pred) - (1-y_true)log(1-y_pred)
梯度下降法求：损失最小时候的系数
"""
data, target = datasets.make_blobs(centers=2)
plt.scatter(data[:, 0], data[:, 1], c=target)

# 定义变量
W = tf.Variable(np.random.randn(2) * 0.02,dtype=tf.float32)
B = tf.Variable(0.0,dtype=tf.float32)
data = tf.constant(data, dtype=tf.float32)
# target = tf.constant(target, dtype=tf.float32)


# 预测函数
def sigmoid(x):
    # 矩阵的乘法
    linear = tf.matmul(x, W) + B
    return tf.nn.sigmoid(linear)


# 定义损失函数
def cross_entropy_loss(y_pred, y_true):
    y_pred = tf.reshape(y_pred, 100)
    # y_pred = tf.clip_by_value(y_pred, 1e-9, 1)
    # 平均损失
    return tf.reduce_mean(
        -(tf.multiply(y_true, tf.math.log(y_pred)) + tf.multiply((1 - y_true), tf.math.log(1 - y_pred))))


# 定义优化器
optimizer = tf.keras.optimizers.SGD()  # 批量梯度下降


# 定义优化过程
def run_optimizers():
    with tf.GradientTape() as g:
        pred = sigmoid(data)  # 模型
        loss = cross_entropy_loss(pred, target)  # 损失函数
    # 计算loss 函数 对[W, B]的梯度
    gradients = g.gradient(loss, [W, B])  # 返回梯度
    # 根据梯度，更新W，B
    optimizer.apply_gradients(zip(gradients, [W, B]))


# 计算准确率
# 计算准确率
def accuracy_score(y_pred, y_true):
    # 概率大于0.1 输出1
    y_pred = tf.reshape(y_pred, 100).numpy()
    y_ = y_pred > 0.5
    print(y_.astype(np.int32))
    print(y_true)
    res = (y_ == y_true)
    return res.mean()


# 训练
for step in range(5000):
    run_optimizers()  # 训练一次
    if 0 == step % 100:
        pred = sigmoid(data)
        acc = accuracy_score(pred, target)
        loss = cross_entropy_loss(pred, target)
        print(f"step:{step} 准确率:{acc} 损失函数:{loss}")
