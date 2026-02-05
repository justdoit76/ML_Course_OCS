import torch
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader

# import dataset
train_mnist = datasets.MNIST(
    root='./data',
    train=True,
    download=True,        
)

# train_ds
print(train_mnist.data.shape)
print(train_mnist.targets.shape)

test_mnist = datasets.MNIST(
    root='./data',
    train=False,
    download=True,        
)

from MNIST_func import plot_MNIST
plot_MNIST(train_mnist, 0, 10)

# data, (60000, 28, 28) :  (0~255) -> (0~1)
X_train = train_mnist.data.float().div(255).flatten(1)
y_train = train_mnist.targets

X_test  = test_mnist.data.float().div(255).flatten(1)
y_test  = test_mnist.targets

train_ds = TensorDataset(X_train, y_train)
test_ds  = TensorDataset(X_test, y_test)

# from google.colab import files
# files.upload()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'using device={device}')

bt_size = 256
use_pin = False

if torch.cuda.is_available():
    bt_size *= 16
    use_pin = True
    print(torch.cuda.get_device_name(0))

train_dl = DataLoader(train_ds, batch_size=bt_size, shuffle=True, pin_memory=use_pin, num_workers=0)
test_dl  = DataLoader(test_ds, batch_size=64)

import torch.nn as nn
m, n = X_train.shape
print(m, n)
C = 10

model = nn.Sequential(    
    nn.Linear(n, C),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 100

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

    if i%10==0:        
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

    accuracy = cnt / len(test_ds)
    print(f'Accuracy={accuracy:.3f}')