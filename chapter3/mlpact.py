import torch as t
import numpy as np
import matplotlib.pyplot as plt
import d2lzh_pytorch as d2l

# def xyplot(x_vals, y_vals, name):
#     d2l.set_figsize(figsize=(5, 2.5))
#     d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
#     d2l.plt.xlabel('x')
#     d2l.plt.ylabel(name +'(x)')
#
# x = t.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = x.relu()
# xyplot(x, y, 'relu')

# 获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

#定义模型
num_inputs, num_outputs, num_hidden = 784, 10 ,256

W1 = t.tensor(np.random.normal(0, 0.01, (num_inputs, num_hidden)), dtype=t.float,
              requires_grad=True)
b1 = t.zeros(num_hidden, dtype=t.float, requires_grad=True)
W2 = t.tensor(np.random.normal(0, 0.01, (num_hidden, num_outputs)), dtype=t.float,
              requires_grad=True)
b2 = t.zeros(num_outputs, dtype=t.float, requires_grad=True)
params = [W1, b1, W2, b2]
# 激活函数
def relu(X):
    return t.max(input=X, other=t.tensor(0.0))

# 定义模型
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(t.matmul(X, W1) +b1)
    return t.matmul(H, W2) + b2

loss = t.nn.CrossEntropyLoss()
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)