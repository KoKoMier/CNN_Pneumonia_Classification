import torchvision
from torch.utils.data import DataLoader
from torch import nn
from model import *
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

classes = ('Covid  ', 'Normal', 'Pneumonia')

# 设置图像预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小
    transforms.ToTensor()           # 转换为Tensor
])

# 加载数据集
dataset = datasets.ImageFolder('archive/Covid19-dataset/train', transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 获取一批数据
data_iter = iter(data_loader)
images, labels = next(data_iter)


# 显示图片的函数
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 显示图片
imshow(torchvision.utils.make_grid(images))
print("类别:", classes[labels[0]])