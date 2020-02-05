import torch as t
from torch import nn

def comp_conv2d(conv2d, X):
    X = X.view((1,1)+X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])


conv2d = nn.Conv2d(in_channels=1, out_channels=1,
                   kernel_size=3, padding=1)

X = t.rand(8,8)
Y = comp_conv2d(conv2d, X)
print(Y)
