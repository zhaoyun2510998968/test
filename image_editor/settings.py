# -*- coding: utf-8 -*-
# @File    : settings.py
# @Author  : AaronJny
# @Time    : 2020/03/13
# @Desc    :

CONTENT_LAYER_WEIGHTS = {'conv4_2': 1.0}
STYLE_LAYER_WEIGHTS = {'conv1_1': 0.2, 'conv2_1': 0.2, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}

CONTENT_LOSS_FACTOR = 1.0
STYLE_LOSS_FACTOR = 1e4
LEARNING_RATE = 0.02
EPOCHS = 3
STEPS_PER_EPOCH = 100
WIDTH = 256
HEIGHT = 256

# 内容特征层及loss加权系数
CONTENT_LAYERS = {'block4_conv2': 0.5, 'block5_conv2': 0.5}
# 风格特征层及loss加权系数
STYLE_LAYERS = {'block1_conv1': 0.2, 'block2_conv1': 0.2, 'block3_conv1': 0.2, 'block4_conv1': 0.2,
                'block5_conv1': 0.2}
# 内容图片路径
CONTENT_IMAGE_PATH = './images/content.jpg'
# 风格图片路径
STYLE_IMAGE_PATH = './images/styles.jpg'
# 生成图片的保存目录
OUTPUT_DIR = './output'

# 内容loss总加权系数
CONTENT_LOSS_FACTOR = 1
# 风格loss总加权系数
STYLE_LOSS_FACTOR = 100

# 图片宽度
WIDTH = 450
# 图片高度
HEIGHT = 300

# 训练epoch数
EPOCHS = 2
# 每个epoch训练多少次
STEPS_PER_EPOCH = 10
# 学习率
LEARNING_RATE = 0.03

# 图像基础设置
WIDTH = 256
HEIGHT = 256

# 损失加权因子
CONTENT_LOSS_FACTOR = 1.0
STYLE_LOSS_FACTOR = 1e4

# 优化参数
LEARNING_RATE = 0.02
EPOCHS = 1
STEPS_PER_EPOCH = 20
