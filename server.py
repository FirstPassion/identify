# -*- coding: utf-8 -*-
# @Time    : 2022-07-20 15:50
# @Author  : Da
# @File    : server.py
import torch
from PIL import Image
from flask import Flask, render_template, request
from model import Net, transform

app = Flask(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files['file']
    name = f.filename
    if name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png'):
        img = Image.open(f).convert('RGB')
        print('正在识别', name)
        return resolutionImg(img)
    return '请上传图片类型的文件'


# 识别图片是猫还是狗
def resolutionImg(img):
    net = Net().to(device)
    net.load_state_dict(torch.load('one.pth'))
    labels = ['猫', '狗']
    img = transform(img)
    img = img.unsqueeze(0)
    p = net(img.to(device))
    p = p.argmax()
    return labels[p]


if __name__ == '__main__':
    app.run('127.0.0.1', 9999, True)
