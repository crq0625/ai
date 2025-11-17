# function print study
# create by gather
# create time 2025/9/16
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)


# 定义网络
class BasicBlock(keras.layers.Layer):
    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super.__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            out_channel,
            kernel_size=3,
            strides=strides,
            padding='same',
            use_bias=False  # 使用BN，就不使用bias
        )
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = keras.layers.Conv2D(
            out_channel,
            kernel_size=3,
            strides=strides,
            padding='same',
            use_bias=False  # 使用BN，就不使用bias
        )
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.relu = keras.layers.ReLU()
        self.add = keras.layers.Add()
        self.downsample = downsample

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)
        x = self.conv1(identity)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.add([identity,x])
        x = self.relu(x)
        return x

class Bottleneck(keras.layers.Layer):
    expansion = 4
    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super.__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(
            out_channel,
            kernel_size=1,
            strides=strides,
            padding='same',
            use_bias=False  # 使用BN，就不使用bias
        )
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = keras.layers.Conv2D(
            out_channel,
            kernel_size=3,
            strides=strides,
            padding='same',
            use_bias=False  # 使用BN，就不使用bias
        )
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.conv3 = keras.layers.Conv2D(
            out_channel * self.expansion,
            kernel_size=1,
            strides=strides,
            padding='same',
            use_bias=False  # 使用BN，就不使用bias
        )
        self.bn3 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.relu = keras.layers.ReLU()
        self.add = keras.layers.Add()
        self.downsample = downsample

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)
        x = self.conv1(identity)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([identity,x])
        x = self.relu(x)
        return x