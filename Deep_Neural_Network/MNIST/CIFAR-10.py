from torchvision.datasets import CIFAR10
from torchvision import transforms

tf = transforms.ToTensor()

cifar_train = CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=tf
)

cifar_test = CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=tf
)

print(cifar_train.data.shape)
print(len(cifar_train.targets))
# transforms 은 꺼내올 때 바뀜, 즉시바뀜(X)
img, label = cifar_train[0]
print(img.shape, label)

# from MNIST_func import plot_CIFAR10
# plot_CIFAR10(cifar_train, 0, 20)

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
bt_size = 256
pin_mode = False
print(f'device={device}')

if torch.cuda.is_available():
    bt_size *= 2
    pin_mode = True
    print(f'gpu={torch.cuda.get_device_name()}')

train_dl = DataLoader(cifar_train, batch_size=bt_size, shuffle=True, pin_memory=pin_mode)

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1), # (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), # (16, 16, 16)

            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1), # (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # (32, 8, 8)

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1), # (64, 8, 8)
            nn.BatchNorm2d(64),
            nn.ReLU(),            
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),      
            nn.Dropout(p=0.5),      
            nn.Linear(128, 10)
        )

    def forward(self, X):
        X = self.conv(X)
        X = self.fc(X)
        return X
    
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( model.parameters(), lr=0.001)
epochs = 20

for i in range(epochs):
    model.train()
    cost = 0

    for x_batch, y_batch in train_dl:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss.item()

    cost /= len(train_dl)
    print(f'epoch={i}, cost={cost:.3f}')

# predict
model.eval()
with torch.no_grad():
    test_dl = DataLoader(cifar_test, batch_size=128)
    cnt = 0

    for x_batch, y_batch in test_dl:        
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        Z = model(x_batch)
        A = torch.argmax(Z, dim=1)

        cnt += (A==y_batch).sum().item()

    acc = cnt / len(cifar_test)
    print(f'Accuracy={acc:.3f}')