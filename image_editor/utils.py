# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : AaronJny
# @Time    : 2020/03/13
# @Desc    :
import tensorflow as tf
import settings
import numpy as np
from PIL import Image

# 我们准备使用经典网络在imagenet数据集上的与训练权重，所以归一化时也要使用imagenet的平均值和标准差
image_mean = tf.constant([0.485, 0.456, 0.406])
image_std = tf.constant([0.299, 0.224, 0.225])


def normalization(x):
    """
    对输入图片x进行归一化，返回归一化的值
    """
    return (x - image_mean) / image_std


def load_images(image_path, width=settings.WIDTH, height=settings.HEIGHT):
    """
    加载并处理图片
    :param image_path:　图片路径
    :param width: 图片宽度
    :param height: 图片长度
    :return:　一个张量
    """
    # 加载文件
    x = tf.io.read_file(image_path)
    # 解码图片
    x = tf.image.decode_jpeg(x, channels=3)
    # 修改图片大小
    x = tf.image.resize(x, [height, width])
    x = x / 255.
    # 归一化
    x = normalization(x)
    x = tf.reshape(x, [1, height, width, 3])
    # 返回结果
    return x


def save_image(image, filename):
    x = tf.reshape(image, image.shape[1:])
    x = x * image_std + image_mean
    x = x * 255.
    x = tf.cast(x, tf.int32)
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)
    x = tf.image.encode_jpeg(x)
    tf.io.write_file(filename, x)

def preprocess_image(pil_image, target_height=settings.HEIGHT, target_width=settings.WIDTH):
    """
    将 PIL 图像缩放、归一化，并转换为模型可用张量
    """
    pil_image = pil_image.resize((target_width, target_height))
    img_array = np.array(pil_image).astype(np.float32) / 255.0
    img_tensor = tf.convert_to_tensor(img_array)
    img_tensor = normalization(img_tensor)
    img_tensor = tf.reshape(img_tensor, [1, target_height, target_width, 3])
    return img_tensor


def tensor_to_image(tensor):
    """
    将张量转换为 PIL 图像
    """
    tensor = tf.reshape(tensor, tensor.shape[1:])  # 去掉 batch 维度
    tensor = tensor * image_std + image_mean
    tensor = tensor * 255.0
    tensor = tf.clip_by_value(tensor, 0, 255)
    tensor = tf.cast(tensor, tf.uint8)
    return Image.fromarray(tensor.numpy())