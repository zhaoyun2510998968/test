from PIL import Image, ImageEnhance


def apply_filter(image_path):
    # 打开图像
    img = Image.open(image_path)

    # 例如，简单地调整亮度
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.5)  # 增加亮度

    # 保存处理后的图像
    filtered_image_path = image_path.replace('uploads', 'uploads/filtered')
    img.save(filtered_image_path)

    return filtered_image_path
