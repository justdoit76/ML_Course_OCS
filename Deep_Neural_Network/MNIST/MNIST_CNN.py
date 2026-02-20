from torchvision.datasets import MNIST

mnist_train = MNIST(
    root='./data',
    train=True,
    download=True
)

mnist_test = MNIST(
    root='./data',
    train=False,
    download=True
)

#from MNIST_func import plot_MNIST
#plot_MNIST(mnist_train, 0, 20)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

X_train = mnist_train.data.float().div(255).unsqueeze(1)
y_train = mnist_train.targets

# cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bt_size = 256
pin_mode = False

print(f'device={device}')

if torch.cuda.is_available():
    bt_size*=16
    pin_mode = True
    print(f'cuda={torch.cuda.get_device_name()}')

train_ds = TensorDataset(X_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bt_size, shuffle=True, pin_memory=pin_mode)

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2), # (16, 14, 14)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2), # (32, 7, 7)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, X):
        X = self.conv(X)
        X = self.fc(X)
        return X
    

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10

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
    X_test = mnist_test.data.float().div(255).unsqueeze(1)
    y_test = mnist_test.targets

    test_ds = TensorDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=256, pin_memory=pin_mode)

    cnt=0

    for x_batch, y_batch in test_dl:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        Z = model(x_batch)
        A = torch.argmax(Z, dim=1)        

        cnt += (A==y_batch).sum().item()

    acc = cnt / len(test_ds)
    print(f'Accuracy={acc:.3f}')


