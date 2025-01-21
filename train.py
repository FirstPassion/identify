# -*- coding: utf-8 -*-
# @Time    : 2022-07-20 10:59
# @Author  : Da
# @File    : train.py
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from model import Net, transform

# 超参数配置
config = {
    "device": torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),  # 选择设备：GPU或CPU
    "train_data_path": "./data",  # 训练数据路径
    "batch_size": 4,  # 批量大小
    "learning_rate": 1e-4,  # 学习率
    "num_epochs": 100,  # 训练轮数
    "patience": 10  # 早停的耐心值
}

# 检查训练数据是否存在
if not os.path.exists(config["train_data_path"]):
    print("训练数据不存在")
    exit()

# 加载训练集
train_data = torchvision.datasets.ImageFolder(
    root=config["train_data_path"], transform=transform
)  # 加载图像数据集
train_data_loader = DataLoader(
    train_data, batch_size=config["batch_size"]
)  # 创建数据加载器

net = Net()
if os.path.exists("best_model.pth"):
    net.load_state_dict(torch.load("best_model.pth"))
    net.to(config["device"])
    print("加载最优模型")
else:
    net.to(config["device"])  # 初始化网络并将其移动到指定设备

# 损失函数
criterion = nn.CrossEntropyLoss().to(config["device"])  # 使用交叉熵损失函数

# 优化器
optimizer = optim.Adam(net.parameters(), lr=config["learning_rate"])  # 使用Adam优化器

# 用于记录损失值
losses = []  # 存储每个epoch的损失值
best_loss = float("inf")  # 初始化最优损失值为无穷大
patience_counter = 0  # 初始化耐心计数器


def calculate_accuracy(data_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            outputs = net(inputs)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 累加总样本数
            correct += (predicted == labels).sum().item()  # 累加正确预测的样本数
    return correct / total  # 返回准确率


def train():
    global best_loss, patience_counter  # 声明使用全局变量best_loss和patience_counter
    for e in range(config["num_epochs"]):  # 遍历每个epoch
        train_loss = 0.0  # 初始化当前epoch的损失
        # 使用 tqdm 显示进度条，并添加当前学习率和损失信息
        with tqdm(total=len(train_data_loader), desc=f"Epoch {e + 1}", unit="batch") as pbar:
            for d in train_data_loader:  # 遍历数据加载器
                x, y = d  # 获取输入数据和标签
                optimizer.zero_grad()  # 清空梯度
                o = net(x.to(config["device"]))  # 前向传播
                loss = criterion(o, y.to(config["device"]))  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                train_loss += loss.data.item()  # 累加损失

                # 更新进度条
                pbar.set_postfix(loss=loss.data.item(), lr=config["learning_rate"])  # 显示当前损失和学习率
                pbar.update(1)  # 更新进度条

        train_loss /= len(train_data_loader)  # 计算平均损失
        losses.append(train_loss)  # 记录损失
        accuracy = calculate_accuracy(train_data_loader)  # 计算训练集上的准确率
        print(f"第{e + 1}次训练的损失值为{train_loss}, 准确率为{accuracy:.2f}")

        # 保存最优模型
        if train_loss < best_loss:  # 如果当前损失小于最优损失
            best_loss = train_loss  # 更新最优损失
            torch.save(net.state_dict(), "best_model.pth")  # 保存最优模型
            print(f"保存最优模型，损失值为{best_loss}")
            patience_counter = 0  # 重置耐心计数器
        else:
            patience_counter += 1  # 增加耐心计数器

        # 检查是否达到早停条件
        if patience_counter >= config["patience"]:
            print(f"早停触发，训练停止。当前最佳损失为{best_loss}")
            break  # 退出训练循环

    plot_losses()  # 绘制损失曲线


def plot_losses():
    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.plot(losses, label="Training Loss")  # 绘制训练损失曲线
    plt.title("Training Loss Over Epochs")  # 设置标题
    plt.xlabel("Epochs")  # 设置x轴标签
    plt.ylabel("Loss")  # 设置y轴标签
    plt.legend()  # 显示图例
    plt.grid()  # 显示网格
    plt.savefig("loss_curve.png")  # 保存损失曲线图像
    plt.show()  # 显示损失曲线


if __name__ == "__main__":
    train()  # 开始训练
