# -*- coding: utf-8 -*-
# @Time    : 2022-07-20 10:59
# @Author  : Da
# @File    : train.py
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader

from model import Net, transform

# 用gpu还是cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_data_path = './data'

# 加载训练集
train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
train_data_loader = DataLoader(train_data, batch_size=4)

net = Net().to(device)

# 损失函数
criterion = nn.CrossEntropyLoss().to(device)

# 优化器
optimizer = optim.Adam(net.parameters(), lr=1e-3)


def train():
    for e in range(1000):
        train_loss = 0.0
        for d in train_data_loader:
            x, y = d
            optimizer.zero_grad()
            o = net(x.to(device))
            loss = criterion(o, y.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.data.item()
        train_loss /= len(train_data_loader)
        print(f"第{e + 1}次训练的损失值为{train_loss}")
    torch.save(net.state_dict(), 'one.pth')


if __name__ == '__main__':
    train()
