import keras.optimizers
from keras.models import Sequential  # 循序模型
from keras.layers import Dense, Activation  # 神经元和激活函数
import keras.backend as K

model = Sequential()
model.add(Dense(32, input_dim=784, activation='relu'))
model.add(1, Activation('sigmoid'))  # 激活函数
model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=[['accuracy']])

# 优化器
# keras.optimizers.RMSprop
# keras.losses.categorical_crossentropy
# 多分类问题
model.compile(optimizer='RMSprop',
              loss='categorical_crossentropy',
              metrics=[['accuracy']]
              )
# 二分类
model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=[['accuracy']]
              )
# 均方误差回归问题
model.compile(optimizer='RMSprop',
              loss='mse')


# 自定义评估标准函数

def mean_pred(y_pred, y_true):
    return K.mean(y_pred)


model.compile(optimizer='RMSprop',
              loss='binary_crossentropy',
              metrics=[['mean_pred']])
