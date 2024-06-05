from PIL import Image
import torchvision
from model import *

device = torch.device('cpu')
# archive\Covid19-dataset\\test\Normal\\0101.jpeg   archive\Covid19-dataset\test\\Viral Pneumonia\\0101.jpeg
image_path= "archive/Covid19-dataset/test/Covid\\0111.jpg" # archive\Covid19-dataset\test\\Covid\\0111.jpg
image = Image.open(image_path)
#print(image)
imgae = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
# print(image.shape)

image = torch.reshape(image,(1,3,32,32))

model = torch.load("./mode/net_264.pth")
# print(model)
model.eval()
model.to(device)

with torch.no_grad():
    output = model(image)
    
# print(output)
# print(output.argmax(1))
predicted_label = output.argmax(1)

# 根据预测结果输出对应的类别名称
if predicted_label == 0:
    print("Covid")
elif predicted_label == 1:
    print("Normal")
elif predicted_label == 2:
    print("Viral Pneumonia")