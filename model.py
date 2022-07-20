# -*- coding: utf-8 -*-
# @Time    : 2022-07-20 15:50
# @Author  : Da
# @File    : model.py
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            Conv2d(3, 128, kernel_size=(3, 3)),
            MaxPool2d(3),
            ReLU(),
            Conv2d(128, 64, kernel_size=(3, 3)),
            MaxPool2d(3),
            ReLU(),
            Conv2d(64, 32, kernel_size=(3, 3)),
            MaxPool2d(3),
            Flatten(),
        )
        self.fc2 = nn.Sequential(
            Linear(32 * 3 * 3, 64),
            ReLU(),
            Linear(64, 2)
        )

    def forward(self, x):
        o = self.fc1(x)
        # print(o.shape)
        o = self.fc2(o)
        # print(o.shape)
        return o


transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
