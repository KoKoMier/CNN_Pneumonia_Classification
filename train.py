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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cup")


# 设置图像预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小
    transforms.ToTensor()           # 转换为Tensor
])

# 加载数据集
train_data = datasets.ImageFolder('archive/Covid19-dataset/train', transform=transform)
test_data = datasets.ImageFolder('archive/Covid19-dataset/test', transform=transform)

#数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))
#利用Dataloader来加载数据
train_dataloader = DataLoader(train_data,batch_size = 64)
test_dataloader = DataLoader(test_data,batch_size =64)
#创建网络模型
mode = mode()
mode = mode.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
optimzier = torch.optim.Adam(mode.parameters())

#设置训练网络的一些参数
## 记录训练的次数
total_train_step = 0
##记录测试的次数
total_test_step = 0
##训练的轮数
epoch = 1000

for i in range(epoch):
    print("---------第{}轮训练开始--------".format(i+1))
    #训练步骤开始
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = mode(imgs)
        loss = loss_fn(outputs,targets)

        #优化器优化模型
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数{},Loss:{}".format(total_train_step,loss.item()))
    #测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = mode(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    total_test_step = total_test_loss + 1
    
    torch.save(mode,"./mode/net_{}.pth".format(i))
    
    print("模型已保存")
