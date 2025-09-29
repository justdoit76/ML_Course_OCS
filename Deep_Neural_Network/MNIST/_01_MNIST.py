import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# load data for mean, std, ToTensor() 0~200 -> 0.0~1.0
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
# 몇 개를 시각화할지 설정
num = 20
col = 5
row = (num + col - 1) // col

plt.figure(figsize=(10, 6))

for i in range(num):
    plt.subplot(row, col, i + 1)
    plt.imshow(dataset[i][0][0].numpy(), cmap='gray')
    plt.title(f"Label: {dataset[i][1]}")
    plt.axis('off')

plt.tight_layout()
plt.show()


num, w, h = dataset.data.size()
loader  = DataLoader(dataset, num, shuffle=False)


# 표준 정규화 (Z-score normalization)

itr = iter(loader)
data,_ = next(itr) # data [60000, 1, 28, 28],  label [60000]을 리턴

mean = data.mean().item()
std  = data.std().item()
print(f"Mean: {mean:.4f}, Std: {std:.4f}")
del dataset

# reload dataset
transform = transforms.Compose([
    transforms.ToTensor(), # 0~255 -> 0~1
    transforms.Normalize((mean, ), (std, )) # 평균, 표준편차 정규화
])

train_ds = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_ds = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
test_dl  = DataLoader(test_ds, batch_size=128, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1   = nn.Linear(64*7*7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool( F.relu( self.conv1(x) ) )
        x = self.pool( F.relu( self.conv2(x) ) )
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu( self.fc1(x) )
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x_batch, y_batch in train_dl:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(x_batch)
        loss = criterion(pred, y_batch)       

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()        
        _, predicted = torch.max(pred, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    acc = correct / total * 100

    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.2f}%")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x_batch, y_batch in test_dl:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(x_batch)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

test_acc = correct / total * 100
print(f"Test Accuracy: {test_acc:.2f}%")