# -*- coding: utf-8 -*-
# @Time    : 2022-07-20 15:50
# @Author  : Da
# @File    : model.py
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU
from torchvision import transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            # 输入通道数为3，输出通道数为128，卷积核大小为3x3，padding为1
            # 输入尺寸: (3, 128, 128) -> 输出尺寸: (128, 128, 128)
            Conv2d(3, 128, kernel_size=(3, 3), padding=1),  
            # 最大池化层，池化窗口大小为2x2
            # 输出尺寸: (128, 128) -> (128, 64, 64)
            MaxPool2d(2),  
            ReLU(),
            # 输入通道数为128，输出通道数为64
            # 输入尺寸: (128, 64, 64) -> 输出尺寸: (64, 64, 64)
            Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            MaxPool2d(2),  # 输出尺寸: (64, 64) -> (64, 32, 32)
            ReLU(),
            # 输入通道数为64，输出通道数为32
            # 输入尺寸: (64, 32, 32) -> 输出尺寸: (32, 32, 32)
            Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            MaxPool2d(2),  # 输出尺寸: (32, 32) -> (32, 16, 16)
            Flatten(),  # 将特征图展平
        )
        self.fc2 = nn.Sequential(
            # 输入特征数为32 * 16 * 16，输出特征数为64
            # 输入尺寸: (32, 16, 16) -> 输出尺寸: (64)
            Linear(32 * 16 * 16, 64),  
            ReLU(),
            # 输出特征数为2
            Linear(64, 2)  
        )

    def forward(self, x):
        o = self.fc1(x)
        # print(o.shape)
        o = self.fc2(o)
        # print(o.shape)
        return o


transform = transforms.Compose([
    transforms.Resize(128),  # 将输入图像调整为128x128
    transforms.CenterCrop((128, 128)),  # 中心裁剪
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])
