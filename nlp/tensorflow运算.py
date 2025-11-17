# function print study
# create by gather
# create time 2025/8/30
import tensorflow as tf
import numpy as np

print(tf.__version__)  # 查看tensorflow 版本
print(tf.config.list_physical_devices('GPU'))  # 查看gpu数量
# 常量
a = tf.constant(1)
print(a)
b = tf.constant([[1, 2, 3], [4, 5, 6]])
b
b.numpy()
tf.square(a)

b = np.random.randint(1, 20, size=(3, 5))
tf.constant(b)
a = tf.constant('abcd')
a
tf.strings.length(a)
t = tf.constant(['coff', 'caf', '咖啡'])
tf.strings.length(t, unit="UTF-8")
# 拼接
r1 = tf.ragged.constant([[1, 2], [3, 4, 5], [6]])
r2 = tf.ragged.constant([[11, 22], [33, 44, 55], [55]])
tf.concat([r1, r2], axis=0)

# 稀疏矩阵,矩阵中0的元素多于非零的元素
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]], values=[1, 2, 3], dense_shape=(3, 4))
# 变成稠密矩阵
tf.sparse.to_dense(s)
# 矩阵的乘法
ss = tf.constant([[1, 2], [1, 2], [1, 2], [1, 2]])  # 密集张量
tf.sparse.sparse_dense_matmul(s, ss)

# 变量
v = tf.Variable([[1, 2, 3], [4, 5, 6]])
v.numpy
v.value()
v.assign(v * 2)  # assign 分配任务
# 变量的赋值
v[0, 1].assign(5)
v[1].assign([7, 8, 9])
# 运算
a = tf.constant(1)
b = tf.constant(2)
c = tf.constant(3)
a + b
a - b
tf.add(a, b)  # 加
tf.subtract(a, b)  # 减
tf.multiply(a, b)  # 乘
tf.divide(a, b)  # 除法
n = np.random.randint(0, 10, size=(3, 4))
n.sum(axis=0)
tf.reduce_sum(n,axis=0)
# 矩阵的乘法
x = np.random.randint(1,4,size=(3,5))
y = np.random.randint(3,6,size=(5,3))
tf.matmul(x,y)