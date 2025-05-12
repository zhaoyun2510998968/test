import base64
from io import BytesIO
from flask import Flask, request, render_template, send_file, jsonify
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from flask import Flask, request, jsonify, send_file
from PIL import Image
from io import BytesIO
import base64
import tensorflow_hub as hub
import io
from PIL import Image, ImageOps
import os
import tensorflow as tf
from model import NeuralStyleTransferModel
import settings
import utils

app = Flask(__name__,static_folder='image_editor/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop')
def crop_page():
    return render_template('crop.html')


@app.route('/crop_image', methods=['POST'])
def crop_image_visual():
    try:
        data = request.get_json()
        image_data = data['imageData'].split(',')[1]  # 去掉 base64 前缀
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png')

    except Exception as e:
        print("裁剪错误：", e)
        return jsonify({'error': str(e)}), 400

@app.route('/resize_image', methods=['POST'])
def resize_image():
    try:
        data = request.get_json()
        image_data = data['imageData'].split(',')[1]
        scale = int(data['scale'])

        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # 原始尺寸
        width, height = image.size
        new_width = int(width * scale / 100)
        new_height = int(height * scale / 100)

        resized = image.resize((new_width, new_height), Image.LANCZOS)

        buffer = BytesIO()
        resized.save(buffer, format='PNG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/png')

    except Exception as e:
        print("缩放图片出错：", e)
        return jsonify({'error': str(e)}), 400
# 其他功能页面可以类似添加



@app.route('/filter')
def filter_page():
    return render_template('filter.html')

@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    try:
        # 从前端获取JSON数据
        data = request.get_json()
        image_data = data['imageData'].split(',')[1]  # 去掉base64前缀
        filter_type = data['filterType']
        brightness = float(data['brightness'])  # 亮度调整值
        saturation = float(data['saturation'])  # 饱和度调整值

        # 将图像从base64解码为PIL图像
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')

        # 根据不同的滤镜类型处理图像
        if filter_type == 'grayscale':
            image = ImageOps.grayscale(image)
        elif filter_type == 'invert':
            image = ImageOps.invert(image)
        elif filter_type == 'blur':
            image = image.filter(ImageFilter.BLUR)
        elif filter_type == 'brightness':
            # 调整亮度
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        elif filter_type == 'saturation':
            # 调整饱和度
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        elif filter_type == 'initial':
            # 使用默认亮度和饱和度调整
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        else:
            return jsonify({'error': '未知的滤镜类型'}), 400

        # 将处理后的图像保存到内存
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        # 返回图像文件
        return send_file(buffer, mimetype='image/png')

    except Exception as e:
        print("滤镜处理错误：", e)
        return jsonify({'error': str(e)}), 400


model = NeuralStyleTransferModel()
# 页面：风格迁移选择页面
@app.route('/style_transfer', methods=['GET'])
def style_transfer_page():
    return render_template('style_transfer.html')

@app.route('/use_built_in_style')
def use_built_in_style():
    return render_template('style_transfer_system.html')

# 路由：上传自定义风格图
@app.route('/upload_custom_style')
def upload_custom_style():
    return render_template('style_transfer_use.html')

def apply_style_transfer_fast(content_image, style_image):
    # 加载预训练模型
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    # 预处理（[1,H,W,3]、归一化）
    content_tensor = utils.preprocess_image(content_image)
    style_tensor = utils.preprocess_image(style_image)

    # 风格迁移（仅一次前向）
    stylized_image = hub_model(tf.constant(content_tensor), tf.constant(style_tensor))[0]

    return utils.tensor_to_image(stylized_image)
def apply_style_transfer(content_image, style_image):
    """
    对输入的 content_image 和 style_image 执行风格迁移，返回 PIL.Image 对象。
    要求 model(image)['content'] 和 ['styles'] 均返回 List[(Tensor, weight)]
    """
    try:
        # === Step 1: 预处理图像 ===
        content_tensor = utils.preprocess_image(content_image)
        style_tensor = utils.preprocess_image(style_image)

        # === Step 2: 提取目标特征 ===
        target_content_features = model([content_tensor])['content']  # [(tensor, weight)]
        target_style_features = model([style_tensor])['styles']       # [(tensor, weight)]

        # === Step 3: 初始化生成图像（用内容图） ===
        noise_image = tf.Variable(tf.identity(content_tensor))  # 可选：加噪声 tf.random.uniform(...)

        # === Step 4: 优化器 ===
        optimizer = tf.keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE)

        # === Step 5: Gram 矩阵函数（高效版） ===
        def gram_matrix(feature):
            x = tf.reshape(feature, [-1, feature.shape[-1]])  # [H*W, C]
            return tf.matmul(x, x, transpose_a=True)

        # === Step 6: 内容损失 ===
        def compute_content_loss(gen_content):
            loss = 0.0
            for (gen_feat, weight), (target_feat, _) in zip(gen_content, target_content_features):
                loss += tf.reduce_mean(tf.square(gen_feat - target_feat)) * weight
            return loss

        # === Step 7: 风格损失 ===
        def compute_style_loss(gen_style):
            loss = 0.0
            for (gen_feat, weight), (target_feat, _) in zip(gen_style, target_style_features):
                gm_gen = gram_matrix(gen_feat)
                gm_tar = gram_matrix(target_feat)
                loss += tf.reduce_mean(tf.square(gm_gen - gm_tar)) * weight
            return loss

        # === Step 8: 总损失函数 ===
        def total_loss(outputs):
            c_loss = compute_content_loss(outputs['content'])
            s_loss = compute_style_loss(outputs['styles'])
            return settings.CONTENT_LOSS_FACTOR * c_loss + settings.STYLE_LOSS_FACTOR * s_loss

        # === Step 9: 单步训练函数 ===
        @tf.function
        def train_one_step():
            with tf.GradientTape() as tape:
                outputs = model(noise_image)
                loss = total_loss(outputs)
            grad = tape.gradient(loss, noise_image)
            optimizer.apply_gradients([(grad, noise_image)])
            return loss

        # === Step 10: Early Stopping ===
        prev_loss = float('inf')
        tolerance = 1e-2
        no_improve = 0
        patience = 10

        for step in range(settings.EPOCHS * settings.STEPS_PER_EPOCH):
            loss = train_one_step()
            loss_val = loss.numpy()

            if abs(prev_loss - loss_val) < tolerance:
                no_improve += 1
                if no_improve >= patience:
                    print(f"✅ Early stopped at step {step}, loss = {loss_val:.4f}")
                    break
            else:
                no_improve = 0

            if step % 20 == 0:
                print(f"[{step}] loss = {loss_val:.4f}")
            prev_loss = loss_val

        # === Step 11: 输出图像 ===
        result_image = utils.tensor_to_image(noise_image)
        return result_image

    except Exception as e:
        print("风格迁移失败：", e)
        raise e


# 接收前端提交的风格迁移请求（Base64）
@app.route('/style_transfer', methods=['POST'])
def style_transfer_post():
    try:
        data = request.get_json()

        # 解码Base64图像
        content_image = base64_to_image(data['contentImage'])
        style_image = base64_to_image(data['styleImage'])

        # 转换成模型所需的格式（1, H, W, 3）
        content_tensor = utils.preprocess_image(content_image)
        style_tensor = utils.preprocess_image(style_image)

        # 提取目标特征
        target_content_features = model([content_tensor])['content']
        target_style_features = model([style_tensor])['styles']

        # 初始化噪声图像
        noise_image = tf.Variable((content_tensor + tf.random.uniform(content_tensor.shape, -0.2, 0.2)) / 2)

        # 优化器
        optimizer = tf.keras.optimizers.Adam(settings.LEARNING_RATE)

        # 损失函数
        def _compute_content_loss(noise_features, target_features):
            M = settings.WIDTH * settings.HEIGHT
            N = 3
            content_loss = tf.reduce_sum(tf.square(noise_features - target_features))
            return content_loss / (2. * M * N)

        def compute_content_loss(noise_content_features):
            losses = []
            for (nf, f), (tf_, _) in zip(noise_content_features, target_content_features):
                losses.append(_compute_content_loss(nf, tf_) * f)
            return tf.reduce_sum(losses)

        def gram_matrix(feature):
            x = tf.transpose(feature, perm=[2, 0, 1])
            x = tf.reshape(x, (x.shape[0], -1))
            return x @ tf.transpose(x)

        def _compute_style_loss(noise_feature, target_feature):
            M = settings.WIDTH * settings.HEIGHT
            N = 3
            return tf.reduce_sum(tf.square(gram_matrix(noise_feature) - gram_matrix(target_feature))) / (4. * (M**2) * (N**2))

        def compute_style_loss(noise_style_features):
            losses = []
            for (nf, f), (tf_, _) in zip(noise_style_features, target_style_features):
                losses.append(_compute_style_loss(nf, tf_) * f)
            return tf.reduce_sum(losses)

        def total_loss(noise_features):
            c_loss = compute_content_loss(noise_features['content'])
            s_loss = compute_style_loss(noise_features['styles'])
            return c_loss * settings.CONTENT_LOSS_FACTOR + s_loss * settings.STYLE_LOSS_FACTOR

        @tf.function
        def train_one_step():
            with tf.GradientTape() as tape:
                noise_outputs = model(noise_image)
                loss = total_loss(noise_outputs)
            grad = tape.gradient(loss, noise_image)
            optimizer.apply_gradients([(grad, noise_image)])
            return loss

        # 训练指定轮数
        for _ in range(settings.EPOCHS * settings.STEPS_PER_EPOCH):
            train_one_step()

        # 转回PIL图像
        result_image = utils.tensor_to_image(noise_image)

        # 返回图片数据
        buffer = io.BytesIO()
        result_image.save(buffer, format='JPEG')
        buffer.seek(0)
        return send_file(buffer, mimetype='image/jpeg')

    except Exception as e:
        print("风格迁移处理失败：", e)
        return jsonify({'error': '风格迁移处理失败'}), 400

STYLE_IMAGES = {
    "vangogh": "static/styles/vangogh.jpg",
    "picasso": "static/styles/picasso.jpg",
    "cyberpunk": "static/styles/cyberpunk.jpg",
    "ink": "static/styles/ink.jpg",
    "sketch": "static/styles/sketch.jpg"
}

@app.route('/style_transfer_use', methods=['POST'])
def style_transfer_use():
    try:
        # 获取前端传来的内容图base64数据和用户选择的风格
        data = request.get_json()
        content_image_base64 = data.get('contentImage')
        selected_style = data.get('selectedStyle')   # 改这里！！

        if not content_image_base64 or not selected_style:
            return jsonify({'error': '缺少内容图或风格选择'}), 400

        # 查找对应的内置风格图
        style_image_path = STYLE_IMAGES.get(selected_style)
        print("Style image path:", style_image_path)
        if not os.path.exists(style_image_path):
            print("风格图片不存在")
        if not style_image_path or not os.path.exists(style_image_path):
            return jsonify({'error': '所选风格不存在'}), 400

        # 解码内容图
        content_image = Image.open(BytesIO(base64.b64decode(content_image_base64.split(',')[1]))).convert('RGB')

        # 加载内置风格图
        style_image = Image.open(style_image_path).convert('RGB')

        # ====== 在这里调用你的风格迁移算法 ======
        result_image = apply_style_transfer(content_image, style_image)
        # =========================================

        # 把处理好的图片转成二进制返回
        buffer = BytesIO()
        result_image.save(buffer, format='PNG')
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png')

    except Exception as e:
        print("风格迁移错误：", e)
        return jsonify({'error': str(e)}), 500

def base64_to_image(base64_data):
    """
    将Base64字符串转换为PIL图像
    """
    header, encoded = base64_data.split(",", 1)
    data = base64.b64decode(encoded)
    return Image.open(io.BytesIO(data)).convert('RGB')


if __name__ == '__main__':
    app.run(debug=True)
