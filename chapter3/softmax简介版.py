import torch as t
import torchvision
from torch import nn
from torch.nn import init
import numpy as np
import sys
import d2lzh_pytorch as d2l
import torchvision.transforms as transforms

# 读取小批量, 获取数据
# 载入FashionMNIST数据集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/MNIST', train=True,
                                                download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/MNIST', train=False,
                                               download=False, transform=transforms.ToTensor())

batch_size = 256
if sys.platform.startswith('win'):
    num_worker = 0
else:
    num_worker = 4

train_iter = t.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                     num_workers=num_worker)
test_iter = t.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False,
                                    num_workers=num_worker)

num_inputs = 784
num_outputs = 10

# 定义模型
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x):
        y = self.linear(x.shape[0], -1)
        return y


# 对x 的形状进行放平
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

from collections import OrderedDict
# 定义网络模型
net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
         ('linear', nn.Linear(num_inputs, num_outputs))
    ])

)

# 初始化 W, b
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

loss = nn.CrossEntropyLoss()
optimzer = t.optim.SGD(net.parameters(), lr=0.1)

# 训练模型
num_epochs = 5

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
            n+=y.shape[0]

        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train_acc %.3f, test_acc %.3f' %(epoch+1, train_l_sum/n,
                                                                     train_acc_sum/n, test_acc))

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimzer)