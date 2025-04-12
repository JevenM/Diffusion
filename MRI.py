import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib
import random

print(matplotlib.get_backend())  # 查看当前后端
matplotlib.use("Agg")  # 使用非交互式后端
from PIL import Image

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签和图片大小
labels_txt = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
image_size = 150

# 图像预处理
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
)

# 加载数据
X_data = []
Y_data = []

for phase in ["Training", "Testing"]:
    for idx, label in enumerate(labels_txt):
        folder = os.path.join("Brain-Tumor-Classification-DataSet-master", phase, label)
        for img_name in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, img_name))
            img = cv2.resize(img, (image_size, image_size))
            X_data.append(img)
            Y_data.append(idx)

X_data = np.array(X_data)
Y_data = np.array(Y_data)
X_data, Y_data = shuffle(X_data, Y_data, random_state=101)

# 划分训练和测试集
split_ratio = 0.8
total_samples = len(X_data)
train_size = int(split_ratio * total_samples)
test_size = total_samples - train_size

X_train, X_test = X_data[:train_size], X_data[train_size:]
y_train, y_test = Y_data[:train_size], Y_data[train_size:]


# 自定义数据集类
class BrainTumorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


train_dataset = BrainTumorDataset(X_train, y_train, transform)
test_dataset = BrainTumorDataset(X_test, y_test, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 构建模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 9 * 9, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
epochs = 100

for epoch in range(epochs):
    model.train()
    train_loss = 0
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_acc)

    # 验证
    model.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    val_losses.append(val_loss / len(test_loader))
    val_accuracies.append(val_acc)

    print(
        f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val_Loss: {val_loss:.4f}, Val_Acc: {val_acc:.4f}"
    )

# 绘制损失曲线
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("trainloss.pdf", dpi=300, bbox_inches="tight")
plt.clf()

# 准确率曲线
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("trainaccuracy.pdf", dpi=300, bbox_inches="tight")
plt.clf()

# 模型评估
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

# 输出评估指标
print("Accuracy:", accuracy_score(y_true, y_pred) * 100)
print("Precision:", precision_score(y_true, y_pred, average="weighted"))
print("Recall:", recall_score(y_true, y_pred, average="weighted"))
print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print(cm)
plt.figure(figsize=(8, 8))
tick_marks = np.arange(len(labels_txt))
plt.xticks(tick_marks, labels_txt, rotation=45)
plt.yticks(tick_marks, labels_txt)

# 使用 'Blues' 颜色映射，保存为PDF的时候有问题，方块会变成菱形，解决方法：
# 1.要么使用png保存，PDF会导致混淆矩阵的每个方格倾斜填充，很奇怪，Y轴从上到下已经是class0,class1,class2的顺序，不用加[::-1]逆序，但是不能保存为pdf，显示有斜条纹，但是png就可以
# plt.imshow(cm, cmap="Blues", interpolation="nearest", aspect="equal")  # 添加 cmap 参数
# plt.yticks(tick_marks, labels_txt)

# 2.要么使用 pcolormesh 替代 imshow，这种会导致Y轴从上到下不是class0,class1,class2的顺序，解决方法：plt.yticks(tick_marks, labels_txt[::-1])增加那个[::-1]
# 强制将Y轴文字逆序，但是还有个问题，cm也会变成按照第二维度逆序的，所以也需要将cm也按照列逆序一下
X, Y = np.meshgrid(np.arange(cm.shape[1]), np.arange(cm.shape[0]))
cm = cm[::-1]
plt.pcolormesh(X, Y, cm, cmap="Blues", linewidth=0.5)  # edgecolors='k',
plt.yticks(tick_marks, labels_txt[::-1])

# 添加颜色条
plt.colorbar()

# 在单元格中添加数值，并动态设置文本颜色
thresh = cm.max() / 2  # 用于判断文本颜色的阈值
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(
        j,
        i,
        f"{cm[i, j]}",
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black",  # 动态设置颜色
    )

plt.title("Confusion Matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.gcf().canvas.draw()
# plt.imshow，保存为png，使用pcolormesh，可以保存为pdf
plt.savefig("confusionmatrix.pdf", format="pdf", bbox_inches="tight", dpi=300)


# 单图像预测
img_path = (
    "Brain-Tumor-Classification-DataSet-master/Testing/pituitary_tumor/image(20).jpg"
)
img = cv2.imread(img_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (150, 150))
img_tensor = transform(img_resized).unsqueeze(0).to(device)

plt.imshow(img)
plt.title("Input Image")
plt.savefig("example.png")

model.eval()
with torch.no_grad():
    output = model(img_tensor)
    probs = F.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    pred_prob = torch.max(probs).item() * 100

print(f"The predicted class of the input image is: {labels_txt[pred_class]}")
print(f"The probability value of prediction is: {pred_prob:.2f}%")
