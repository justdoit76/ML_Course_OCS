import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

tf = transforms.ToTensor()

# import dataset
train_mnist = datasets.MNIST(
    root='./data',
    train=True,
    transform=tf,
    download=True,        
)

# train_ds
print(train_mnist.data.shape)
print(train_mnist.targets.shape)

test_mnist = datasets.MNIST(
    root='./data',
    train=False,
    transform=tf,
    download=True,        
)

# from google.colab import files
# files.upload()

from MNIST_func import plot_MNIST
plot_MNIST(train_mnist, 0, 10)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device={device}')

bt_size = 128
use_pin = False

if torch.cuda.is_available():
    bt_size *= 2
    use_pin = True
    print(torch.cuda.get_device_name(0))

train_dl = DataLoader(train_mnist, batch_size=bt_size, shuffle=True, pin_memory=use_pin, num_workers=0)
test_dl  = DataLoader(test_mnist, batch_size=64)

import torch.nn as nn
C = 10
BS, H, W = train_mnist.data.shape
model = nn.Sequential(   
    nn.Flatten(), 
    nn.Linear(H*W, C),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 15

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
    cnt = 0
    for x_batch, y_batch in test_dl:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        Z = model(x_batch)
        A = Z.argmax(dim=1)

        cnt += (A==y_batch).sum().item()

    accuracy = cnt / len(test_mnist)
    print(f'Accuracy={accuracy:.3f}')