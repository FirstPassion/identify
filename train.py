# -*- coding: utf-8 -*-
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
import logging
from model import Net, transform

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 设置 Matplotlib 支持中文
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置默认字体为 SimHei（黑体）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class DataLoaderModule:
    """
    数据加载模块：负责加载和预处理数据
    """

    def __init__(self, data_path, batch_size, transform):
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = transform

    def load_data(self):
        """
        加载训练数据、验证数据和测试数据
        """
        train_data = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_path, "train"), transform=self.transform
        )
        val_data = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_path, "val"), transform=self.transform
        )
        test_data = torchvision.datasets.ImageFolder(
            root=os.path.join(self.data_path, "test"), transform=self.transform
        )
        train_data_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )
        val_data_loader = DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False
        )
        test_data_loader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=False
        )
        return train_data_loader, val_data_loader, test_data_loader


class Trainer:
    """
    训练模块：负责训练模型，包括损失计算、反向传播、学习率调整等
    """

    def __init__(self, config, train_data_loader, val_data_loader, test_data_loader):
        self.config = config
        self.device = config["device"]
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.net = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config["learning_rate"])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=2
        )
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.test_accuracy = None

        # 加载模型
        if os.path.exists(config["model_save_path"]):
            self.net.load_state_dict(
                torch.load(config["model_save_path"], weights_only=True)
            )
            logging.info("从 {} 加载最优模型".format(config["model_save_path"]))

    def calculate_accuracy(self, data_loader):
        """
        评估模块：计算模型在给定数据加载器上的准确率
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def train(self):
        """
        训练模型
        """
        for epoch in range(self.config["num_epochs"]):
            self.net.train()
            train_loss = 0.0
            train_accuracy = 0.0

            # 使用 tqdm 显示进度条，避免生成新的进度条
            with tqdm(
                self.train_data_loader,
                desc=f"Epoch {epoch + 1}/{self.config['num_epochs']}",
                unit="batch",
                leave=False,
            ) as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    train_accuracy += self.calculate_accuracy(
                        [(inputs, labels)]
                    )  # 计算当前 batch 的准确率
                    pbar.set_postfix(
                        loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"]
                    )

            train_loss /= len(self.train_data_loader)
            train_accuracy /= len(self.train_data_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)

            # 验证集评估
            val_loss, val_accuracy = self.evaluate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            logging.info(
                f"第 {epoch + 1} 轮训练, 训练损失: {train_loss:.4f}, 训练准确率: {train_accuracy:.4f}, "
                f"验证损失: {val_loss:.4f}, 验证准确率: {val_accuracy:.4f}"
            )

            # 更新学习率
            self.scheduler.step(val_loss)

            # 保存最优模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), self.config["model_save_path"])
                logging.info(f"保存最优模型，验证损失: {self.best_loss:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 早停
            if self.patience_counter >= self.config["patience"]:
                logging.info(
                    f"早停触发，训练停止。当前最佳验证损失: {self.best_loss:.4f}"
                )
                # 训练结束后在测试集上评估模型
                self.test_accuracy = self.calculate_accuracy(self.test_data_loader)
                logging.info(f"测试集准确率: {self.test_accuracy:.4f}")
                break

        # 训练结束后在测试集上评估模型
        self.test_accuracy = self.calculate_accuracy(self.test_data_loader)
        logging.info(f"测试集准确率: {self.test_accuracy:.4f}")

        self.plot_losses()

    def evaluate(self):
        """
        评估模块：在验证集上评估模型性能
        """
        self.net.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in self.val_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                val_accuracy += self.calculate_accuracy([(inputs, labels)])
        val_loss /= len(self.val_data_loader)
        val_accuracy /= len(self.val_data_loader)
        return val_loss, val_accuracy

    def plot_losses(self):
        """
        日志和可视化模块：绘制损失和准确率曲线
        """
        plt.figure(figsize=(12, 5))

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="训练损失")
        plt.plot(self.val_losses, label="验证损失")
        plt.title("训练和验证损失随轮次变化")
        plt.xlabel("轮次")
        plt.ylabel("损失")
        plt.legend()
        plt.grid()

        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label="训练准确率")
        plt.plot(self.val_accuracies, label="验证准确率")
        plt.title("训练和验证准确率随轮次变化")
        plt.xlabel("轮次")
        plt.ylabel("准确率")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig("training_metrics.png")
        plt.show()


def main():
    """
    主程序模块：解析命令行参数并启动训练
    """
    parser = argparse.ArgumentParser(description="训练模型")
    parser.add_argument(
        "--train_data_path", type=str, default="./data", help="训练数据路径"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--patience", type=int, default=10, help="早停")
    parser.add_argument(
        "--model_save_path", type=str, default="best_model.pth", help="模型保存路径"
    )
    args = parser.parse_args()

    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "train_data_path": args.train_data_path,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "model_save_path": args.model_save_path,
    }

    # 加载数据
    data_loader_module = DataLoaderModule(
        config["train_data_path"], config["batch_size"], transform
    )
    train_data_loader, val_data_loader, test_data_loader = (
        data_loader_module.load_data()
    )

    # 训练模型
    trainer = Trainer(config, train_data_loader, val_data_loader, test_data_loader)
    trainer.train()


if __name__ == "__main__":
    main()
