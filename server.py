# -*- coding: utf-8 -*-
# @Time    : 2022-07-20 15:50
# @Author  : Da
# @File    : server.py
from flask import Flask, request, jsonify, render_template  # 导入render_template
import torch
from model import Net  # 导入模型
from PIL import Image
from model import transform
import io

app = Flask(__name__)
model = Net()  # 实例化模型
model.load_state_dict(torch.load('best_model.pth'))  # 加载训练好的模型参数
model.eval()  # 设置模型为评估模式

@app.route('/')
def index():
    return render_template('index.html')  # 渲染index.html页面

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    # 读取图像并进行预处理
    img = Image.open(io.BytesIO(file.read()))
    img = transform(img).unsqueeze(0)  # 增加一个维度以适应模型输入

    with torch.no_grad():  # 不计算梯度
        output = model(img)  # 模型预测
        probabilities = torch.softmax(output, dim=1)  # 计算概率
        confidence, predicted = torch.max(probabilities, 1)  # 获取预测结果和置信度

    # 设置置信度阈值
    confidence_threshold = 0.5  # 置信度阈值
    if confidence.item() < confidence_threshold:
        return jsonify({'predicted_class': '未知', 'confidence': confidence.item()})  # 置信度低于阈值时返回未知

    return jsonify({'predicted_class': predicted.item(), 'confidence': confidence.item()})  # 返回预测结果和置信度

if __name__ == '__main__':
    app.run(debug=True)  # 启动Flask应用
