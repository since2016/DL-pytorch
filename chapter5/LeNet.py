import time
import torch as t
from torch import nn, optim

import sys
sys.path.append("..")
from d2lzh_pytorch import *
device = t.device('cuda' if t.cuda.is_available()
                  else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),   # mnist 每张图片 28x28 ，所以四周有2位填充
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(16, 120, 5),
            # nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Linear(400, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
            nn.Sigmoid()
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(feature.size(0), -1))
        return output

# net = LeNet()
# print(net)

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device

    acc_sum, n = 0.0, 0

    with torch.no_grad():   # 不去计算梯度
        for X, y in data_iter:
            if isinstance(net, t.nn.Module):
                net.eval()
                acc_sum += (net(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1)==y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) ==y).float().sum().item()
            n += y.shape[0]

    return acc_sum/n

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("trianing on", device)
    loss = t.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            # print(X.shape)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d,  loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec" %(
            epoch+1, train_l_sum/batch_count, train_acc_sum/n, test_acc, time.time() - start
        ))
net = LeNet()
lr, num_epochs = 0.001, 5
optimizer = t.optim.Adam(net.parameters(), lr =lr)
train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
