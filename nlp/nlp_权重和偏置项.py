import torch
import numpy as np

# 假数据：batch_size=4, 特征维度=20, 类别数=10
X = torch.randn(4, 5)    # [batch_size, n_features]
print("X shape:", X.shape)
# X shape: torch.Size([4, 20])

linear = torch.nn.Linear(5, 10, bias=True)
# 输入特征是5，输出类别是10 对应的权重矩阵是 [10, 5]，计算的时候先转置成 [5, 10]，torch框架自己完成
print("权重 W shape:", linear.weight.shape)  
print("偏置 b shape:", linear.bias.shape)     # [10]偏置项就是行向量，计算的时候会自动广播到每个样本