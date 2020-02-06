import time
import torch as t
from torch import nn, optim

import sys
import d2lzh_pytorch as d2l
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)

conv_arch = ((1,1,64), (1,64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
fc_feature = 512*7*7
fc_hidden_units = 4096

def vgg(conv_arch, fc_feature, fc_hidden_units=4096):
    net = nn.Sequential()

    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module('vgg_block_'+str(i+1), vgg_block(num_convs, in_channels, out_channels))

    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(),
                                       nn.Linear(fc_feature, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 10)
                                       ))

    return net

net = vgg(conv_arch, fc_feature, fc_hidden_units)
X = t.rand(1, 1, 224, 224)
#
# for name, blk in net.named_children():
#     X = blk(X)
#     print(name, 'output shape: ', X.shape)

ratio = 8
small_conv_arch = [(1,1,64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio),
                   (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]

net = vgg(small_conv_arch, fc_feature//ratio, fc_hidden_units//ratio)
print(net)

batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

lr , num_epochs = 0.001, 5
optimizer = t.optim.Adam(net.parameters(), lr=lr)
d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)