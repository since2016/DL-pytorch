import torch as t
from torch import nn
from torch.nn import init
import numpy as np
import sys
import d2lzh_pytorch as d2l

num_inputs, num_outputs, num_hidden = 784, 10, 256
drop_prob1 = 0.3
net = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hidden),
    nn.ReLU(),
    nn.Dropout(drop_prob1),
    nn.Linear(num_hidden, num_outputs),

)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = t.nn.CrossEntropyLoss()

optimizer = t.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              batch_size, None, None, optimizer)