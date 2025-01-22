# -*- coding: utf-8 -*-
# @Time    : 2025-01-22 15:50
# @Author  : Da
# @File    : server.py
from flask import Flask, request, jsonify, render_template  # 导入render_template
import torch
from model import Net  # 导入模型
from PIL import Image
from model import transform
import io
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = Flask(__name__)
model = Net()  # 实例化模型
model.load_state_dict(
    torch.load("best_model.pth", weights_only=True)
)  # 加载训练好的模型参数
model.eval()  # 设置模型为评估模式

# 假设有一个类别名称映射表
class_names = {0: "猫", 1: "狗"}

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = ["JPEG", "PNG", "BMP", "GIF", "TIFF", "WEBP"]


@app.route("/")
def index():
    return render_template("index.html")  # 渲染index.html页面


@app.route("/upload", methods=["POST"])
def upload():
    # 检查是否有文件上传
    if "file" not in request.files:
        return jsonify({"error": "没有文件部分"}), 400

    file = request.files["file"]
    # 检查文件是否为空
    if file.filename == "":
        return jsonify({"error": "没有选择文件"}), 400

    try:
        # 读取图像并进行预处理
        img = Image.open(io.BytesIO(file.read()))

        # 检查图片格式是否支持
        if img.format not in SUPPORTED_IMAGE_FORMATS:
            return jsonify({"error": f"不支持的图片格式: {img.format}"}), 400

        # 将图片转换为RGB格式（避免某些格式如PNG的RGBA问题）
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = transform(img).unsqueeze(0)  # 增加一个维度以适应模型输入

        with torch.no_grad():  # 不计算梯度
            output = model(img)  # 模型预测
            probabilities = torch.softmax(output, dim=1)  # 计算概率
            # 获取前2个概率和对应的类别索引
            top_probabilities, top_indices = torch.topk(probabilities, 2)

        # 设置置信度阈值
        confidence_threshold = 0.5
        top_confidence = top_probabilities[0][0].item()  # 最高置信度

        if top_confidence < confidence_threshold:  # 检查最高置信度是否低于阈值
            return jsonify(
                {
                    "predicted_class": "未知",
                    "confidence": f"{top_confidence * 100:.2f}%",
                }
            )

        # 获取类别名称和置信度
        predicted_classes = [
            class_names.get(idx.item(), "未知") for idx in top_indices[0]
        ]
        predicted_confidences = [f"{prob * 100:.2f}%" for prob in top_probabilities[0]]

        # 返回结果
        return jsonify(
            {
                "top_prediction": {
                    "class": predicted_classes[0],
                    "confidence": predicted_confidences[0],
                },
                "second_prediction": {
                    "class": predicted_classes[1],
                    "confidence": predicted_confidences[1],
                },
                "all_predictions": [
                    {
                        "class": predicted_classes[0],
                        "confidence": predicted_confidences[0],
                    },
                    {
                        "class": predicted_classes[1],
                        "confidence": predicted_confidences[1],
                    },
                ],
            }
        )

    except Exception as e:
        logging.error(f"处理文件时发生错误: {str(e)}")
        return jsonify({"error": f"处理文件时发生错误: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)  # 启动Flask应用
