# -*- coding: utf-8 -*-
# @Time    : 2022-07-20 15:50
# @Author  : Da
# @File    : model.py
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import torch


class SelfAttention(nn.Module):
    """
    自注意力机制
    """

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        # 隐藏层大小
        self.hidden_size = hidden_size
        # 查询、键、值的线性变换
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, hidden_size)
        Q = self.query(x)  # (batch_size, seq_len, hidden_size)
        K = self.key(x)  # (batch_size, seq_len, hidden_size)
        V = self.value(x)  # (batch_size, seq_len, hidden_size)
        # 计算注意力分数
        score = Q @ K.transpose(-2, -1)  # (batch_size, seq_len, seq_len)
        # 计算注意力权重
        attention = F.softmax(
            score / (self.hidden_size**0.5), dim=-1
        )  # (batch_size, seq_len, seq_len)
        # 计算注意力加权后的输出
        output = attention @ V  # (batch_size, seq_len, hidden_size)
        return output


class Net(nn.Module):
    def __init__(self, in_channels=3, hidden_size=128, num_classes=2):
        super(Net, self).__init__()
        # 卷积层
        # 输入通道数，输出通道数，卷积核大小，padding
        self.conv = nn.Conv2d(in_channels, hidden_size, kernel_size=(3, 3), padding=1)
        # 自注意力层
        self.self_attention = SelfAttention(hidden_size)
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x.shape: (batch_size, in_channels, height, width)
        # print("输入 x.shape:", x.shape)
        x = self.conv(x)
        # 保存卷积后的特征图
        self.conv_output = x
        # x.shape: (batch_size, hidden_size, height, width)
        # print("卷积后 x.shape:", x.shape)
        x = self.self_attention(x)
        # 残差连接
        x = x + self.conv_output
        # x.shape: (batch_size, hidden_size, height, width)
        # print("自注意力后 x.shape:", x.shape)
        # 需要调整形状以适应全连接层
        x = x.view(
            x.size(0), x.size(1), -1
        )  # (batch_size, hidden_size, height * width)
        x = x.mean(dim=2)  # 对 height 和 width 维度取平均
        # x.shape: (batch_size, hidden_size)
        # print("平均后 x.shape:", x.shape)
        x = self.fc(x)
        # x.shape: (batch_size, num_classes)
        # print("全连接后 x.shape:", x.shape)
        return x


transform = transforms.Compose(
    [
        transforms.Resize(128),  # 将输入图像调整为128x128
        transforms.CenterCrop((128, 128)),  # 中心裁剪
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 归一化
    ]
)

if __name__ == "__main__":
    net = Net()
    x = torch.randn(1, 3, 128, 128)
    print(net(x).shape)
