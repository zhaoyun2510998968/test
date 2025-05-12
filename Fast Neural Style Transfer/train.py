import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. 加载内容图像和风格图像
def image_loader(image_name):
    image = Image.open(image_name)
    loader = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image


content_img = image_loader("C:\\Users\\25109\\Desktop\\风格迁移\\图像风格转换器\\Fast Neural Style Transfer\\content.jpg")
style_img = image_loader("C:\\Users\\25109\\Desktop\\风格迁移\\图像风格转换器\\Fast Neural Style Transfer\\style.jpg")

# 2. 定义风格迁移网络
class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()
        # 使用VGG-19作为特征提取器
        vgg = models.vgg19(pretrained=True).features
        self.features = nn.Sequential(
            *list(vgg.children())[:-1]  # 去掉VGG最后的全连接层
        )
        self.conv1 = nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.conv1(x)
        return x

# 3. 定义内容损失和风格损失
def content_loss(output, target):
    # 保证内容图像和生成图像的特征尺寸一致
    return torch.mean((output - target) ** 2)

def style_loss(output, target):
    # 计算格拉姆矩阵
    gram_output = gram_matrix(output)
    gram_target = gram_matrix(target)
    return torch.mean((gram_output - gram_target) ** 2)

def gram_matrix(x):
    batch_size, channels, height, width = x.size()
    features = x.view(batch_size * channels, height * width)
    gram = torch.mm(features, features.t())
    return gram / (channels * height * width)

# 4. 提取特征
def get_features(image, model):
    layers = ['0', '5', '10', '19', '28']  # 选择VGG-19中的某些层
    features = []
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features.append(x)
    return features

# 5. 初始化网络和优化器
style_transfer_net = StyleTransferNet().cuda()
optimizer = optim.Adam(style_transfer_net.parameters(), lr=0.001)

# 6. 提取内容图像和风格图像的特征
content_features = get_features(content_img.cuda(), style_transfer_net.features)
style_features = get_features(style_img.cuda(), style_transfer_net.features)

# 设置损失的权重
alpha = 1e5  # 内容损失的权重
beta = 1e10  # 风格损失的权重

# 7. 训练风格迁移网络
for epoch in range(500):
    optimizer.zero_grad()

    # 前向传播，生成图像
    generated_img = style_transfer_net(content_img.cuda())

    # 获取生成图像的特征
    generated_features = get_features(generated_img, style_transfer_net.features)

    # 计算内容损失
    c_loss = content_loss(generated_features[0], content_features[0])  # 使用VGG的某一层的特征

    # 计算风格损失
    s_loss = 0
    for gf, sf in zip(generated_features, style_features):
        s_loss += style_loss(gf, sf)

    # 总损失
    total_loss = alpha * c_loss + beta * s_loss
    total_loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item()}")
        # 保存中间生成的图像
        if epoch % 100 == 0:
            generated_img = generated_img.squeeze(0)
            generated_img = generated_img.cpu().detach().numpy().transpose(1, 2, 0)
            generated_img = (generated_img - generated_img.min()) / (generated_img.max() - generated_img.min())  # 归一化
            plt.imshow(generated_img)
            plt.show()
