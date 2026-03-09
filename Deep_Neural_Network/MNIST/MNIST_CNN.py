from torchvision.datasets import MNIST
from torchvision import transforms

tf = transforms.ToTensor()

mnist_train = MNIST(
    root='./data',
    train=True,
    transform=tf,
    download=True
)

mnist_test = MNIST(
    root='./data',
    train=False,
    transform=tf,
    download=True
)

# from google.colab import files
# files.upload()

from MNIST_func import plot_MNIST, plot_MNIST_Neurons
plot_MNIST(mnist_train, 0, 20)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bt_size = 128
pin_mode = False

print(f'device={device}')

if torch.cuda.is_available():
    bt_size*=2
    pin_mode = True
    print(f'cuda={torch.cuda.get_device_name()}')

train_dl = DataLoader(mnist_train, batch_size=bt_size, shuffle=True, pin_memory=pin_mode)

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
    img, label = mnist_test[0]
    print(img.shape)
    plot_MNIST_Neurons(model, img, device)

    test_dl = DataLoader(mnist_test, batch_size=128)
    cnt=0

    for x_batch, y_batch in test_dl:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        Z = model(x_batch)
        A = torch.argmax(Z, dim=1)        

        cnt += (A==y_batch).sum().item()

    acc = cnt / len(mnist_test)
    print(f'Accuracy={acc:.3f}')