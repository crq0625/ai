# function print study
# create by gather
# create time 2025/11/13
import dltools
import torch
import torch.nn.functional as F

"""加载数据"""
# vocab 词汇表
# data_iter
# batch_size=64 批量大小,词元的个数 num_steps=10取10个词元，如果是字符就取10个字符，如果是10个单词，就取10个单词
# 数据是次元对应的小标索引，方便后续转换成one-hot编码
batch_size, num_steps = 32, 35
data_iter, vocab = dltools.load_data_time_machine(batch_size, num_steps)
# one-hot编码处理的时候会把元素索引变成一个one-hot向量，就是把一个元素扩展成一个一维数组
# 因此输入一维数组结果会返回二维数组,最后一维变的数量变成词表大小（也就是分类的个数）
feature = torch.arange(10).reshape(2, 5)
print(feature)
print(feature.T)
res = F.one_hot(feature, num_classes=len(vocab))
print(res.shape)
"""
torch.Size([2, 2])
torch.Size([2, 2, 28])
"""
print('------------------')
b_h = torch.zeros(2)
print(b_h.shape)
print(b_h)
print('------------------')

"""初始化模型参数"""
def get_params(vocab_size, num_hiddens,device):
    num_inputs = num_outputs = vocab_size # 输入输出的维度是词表的大小
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    # 隐藏层的参数
    """
    h代表隐藏层
    q代表输出层
    x代表输入层
    num_inputs 输入数据维度，输入矩阵维度是num_inputs
    num_hiddens 隐藏层输出维度，隐藏层权重矩阵输出维度是num_hiddens
    W_xh，W_hh是共享权重矩阵
    b_h,是隐藏层的偏置项，维度是[num_hiddens]
    """
    # 输入层到隐藏层权重维度,输入参数和W_xh矩阵相乘
    W_xh = normal((num_inputs, num_hiddens)) 
    # 隐藏层到隐藏层权重维度,输出维度是num_hiddens然后又作为下一层的输入维度因此，权重的维度为(num_hiddens, num_hiddens)
    W_hh = normal((num_hiddens, num_hiddens)) 
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层的参数
    W_o = normal((num_hiddens, num_outputs)) # 输出层权重矩阵
    b_o = torch.zeros(num_outputs, device=device) # 输出层偏置项

    """设置变量可以求导"""
    params = [W_xh, W_hh, b_h, W_o, b_o]
    for param in params:
        param.requires_grad_(True)
    return params

# 初始化隐藏状态,返回的是元组
def init_rnn_state(batch_size, num_hiddens, device):
    """初始化隐藏状态
    输出维度是num_hiddens

    """
    return (torch.zeros((batch_size, num_hiddens), device=device),)

# rnn主题结构
def rnn(inputs,state,params):
    # inputs 输入数据的形状（时间步数，批次大小，输入维度）
    W_xh, W_hh, b_h, W_o, b_o = params # 取出参数
    H, = state # 隐藏层状态就是H0的状态
    output = []
    """
    X的shape是(批次大小,词表大小)
    """
    for X in inputs:
        """
        torch.mm矩阵的乘法
        矩阵的加法，要求行列完全相同
        torch.mm(X,W_xh) batch_size,num_hiddens
        torch.mm(H,W_hh) batch_size,num_hiddens
        """
        H = torch.tanh(torch.mm(X,W_xh)+torch.mm(H,W_hh)+b_h)
        Y = torch.mm(H,W_o)+b_o
        output.append(Y)
    return torch.cat(output,dim=0),(H,)

# 包装成类
class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_params,init_state,forward_fn):
        self.vocab_size,self.num_hiddens = vocab_size,num_hiddens
        self.params = get_params(vocab_size,num_hiddens,device)
        self.init_state,self.forward_fn = init_state,forward_fn

    def __call__(self, X, state):
        X = F.one_hot(x.T,self.vocab_size).type(torch.float32) # 输入参数
        return self.forward_fn(X,state,self.params)

    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)


device = dltools.try_gpu()
number_hiddens = 512
net = RNNModelScratch(len(vocab), number_hiddens, device, get_params, init_rnn_state, rnn)
state = net.begin_state(X.shape[0], device)
Y,new_state = net(X.to(device), state)
print(Y.shape)
print(new_state[0].shape)
