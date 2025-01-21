# 学pytorch做的一个简单的小例子

> data目录下的2个目录是猫和狗的图片
> 
> best_model.pth是训练中保存最好的模型

## 安装

```shell
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

## 训练模型

```shell
python train.py
```

训练损失曲线
![训练结果](./loss_curve.png)

## 开启web服务

```shell
python server.py
```
