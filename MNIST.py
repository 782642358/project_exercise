import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import openpyxl
import torch.nn.functional as F

# 定义超参数
BATCH_SIZE = 16  # 每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10 # 训练数据集的轮次

# 构建pipeline,对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(),# 将图片转换成tensor
    transforms.Normalize((0.1307,),(0.3081,)) #标准化图像张量
])

# 下载、加载数据
from torch.utils.data import DataLoader

# 下载数据集
train_set = datasets.MNIST("data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST("data", train=False, download=True, transform=pipeline)

# 加载数据
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

#AlexNetMNIST 模型定义
class AlexNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetMNIST, self).__init__()
        self.features = nn.Sequential(
            # 卷积层 + 激活函数 + 池化层
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            # 激活函数 + Dropout+ 全连接层
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x) # 通过卷积层部分
        x = x.view(-1, 256 * 2 * 2) # 展平特征图
        x = self.classifier(x) # 通过全连接层部分
        return x
# 实例化模型、定义损失函数和优化器
model = AlexNetMNIST().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 记录训练过程中的损失和准确率
loss_history = []
accuracy_history = []

# 训练模型
for epoch in range(EPOCHS):
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()  # 清空梯度
        output = model(data) # 前向传播
        loss = criterion(output, target) # 计算损失
        loss.backward() # 反向传播
        optimizer.step() # 更新权重
        running_loss += loss.item()
        _predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    loss_history.append(epoch_loss)
    accuracy_history.append(epoch_accuracy)
    print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# 测试模型
# 计算Precision、Recall、F1-Score、Confusion Matrix等指标
def test_model_and_compute_metrics(model, test_loader):
    model.eval()  # 设置模型为评估模式
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.view(-1).tolist())
            all_labels.extend(labels.view(-1).tolist())
    return all_preds, all_labels

all_preds, all_labels = test_model_and_compute_metrics(model, test_loader)

# 生成分类报告和混淆矩阵
report = classification_report(all_labels, all_preds, output_dict=True)
conf_mat = confusion_matrix(all_labels, all_preds)

report_df = pd.DataFrame(report).transpose()
conf_mat_df = pd.DataFrame(conf_mat, index=[f'Class {i}' for i in range(len(conf_mat))], columns=[f'Class {i}' for i in range(len(conf_mat))])

#excel保存
file_path = 'E:\\code_project_exes\\2.MNIST\\分类报告.xlsx'

try:
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        report_df.to_excel(writer, sheet_name='分类报告', index=True)
        conf_mat_df.to_excel(writer, sheet_name='混淆矩阵', index=True)

    print(f"实验结果已保存到Excel文件 '{file_path}'")
except Exception as e:
    print(f"保存Excel文件时出错: {e}")

# 可视化损失和准确率
plt.figure(figsize=(12, 6))

# 绘制训练损失曲线 Loss
plt.subplot(1, 2, 1)
plt.plot(loss_history, label="Training Loss", color='red', linestyle='-', linewidth=2, marker='o')
plt.title("Training Loss", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# 绘制训练准确率曲线 Accuracy
plt.subplot(1, 2, 2)
plt.plot(accuracy_history, label="Training Accuracy", color='orange', linestyle='-', linewidth=2, marker='x')  plt.title("Training Accuracy", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

plt.show()
