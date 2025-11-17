import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

"""
线性回归封装
"""
# 加载数据
data = pd.read_csv('./dataset/HR.csv')
# pd.get_dummies(data.part)  转换成one-hot编码,对应的表头就是唯一值
data = data.join(pd.get_dummies(data.part).astype(np.int32), how='left').join(
    pd.get_dummies(data.salary).astype(np.int32), how='left')  # 特征变多
data.drop(columns=['part', 'salary'], inplace=True)
print(data.left.value_counts())
# print(11428 / (11428 + 3572))  # 数据不平衡 SMOTE

Y = torch.from_numpy(data.left.values.reshape(-1, 1)).type(torch.float32)
X_data = data[[c for c in data.columns if c != 'left']].values
X = torch.from_numpy(X_data).type(torch.float32)
print('X.shape:', X.shape, 'Y.shape:', Y.shape)


# 子类写法
class HRModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入20个特征，输出64个特征，相当于有64个神经元
        self.linear1 = nn.Linear(20, 64)  # 64
        self.linear2 = nn.Linear(64, 64)  # 64个神经元
        self.linear3 = nn.Linear(64, 1)  # 1个神经元
        self.activate = nn.ReLU()  # 没有激活函数，完全不更新数据
        self.sigmoid = nn.Sigmoid()  # 输出结果映射到0-1之间

    def forward(self, input_x):
        # 定义前向传播
        x_temp = self.linear1(input_x)
        x_temp = self.activate(x_temp)
        x_temp = self.linear2(x_temp)
        x_temp = self.activate(x_temp)
        x_temp = self.linear3(x_temp)
        x_temp = self.sigmoid(x_temp)
        return x_temp


# DataLoader可以自动分批次取数据

lr = 0.001


def get_model():
    model_temp = HRModel()
    return model_temp, torch.optim.Adam(model_temp.parameters(), lr=lr)


loss_fn = nn.BCELoss()
model, optimizer = get_model()

# 定义训练过程
batch_size = 64
steps = X.shape[0] // batch_size
epochs = 100

hrDataset = TensorDataset(X, Y)
hrDataloader = torch.utils.data.DataLoader(hrDataset, batch_size=batch_size)

for epoch in range(epochs):
    for x, y in hrDataloader:
        y_pred = model(x)  # 预测值
        loss = loss_fn(y_pred, y)  # 损失函数值
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度
    # print('epoch: {}, loss: {}'.format(epoch, loss_fn(model(X), Y)))

print('准确率', ((model(X).data.numpy() > 0.5) == Y.data.numpy()).mean())
