from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=500, noise=0.2)
X = X.astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

plt.scatter(X[:,0], X[:, 1], c=y[:, 0], cmap='coolwarm', alpha=0.5)
plt.show()

import torch
from torch.utils.data import TensorDataset, DataLoader

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

tensor_ds = TensorDataset(X_tensor, y_tensor)
tensor_dl = DataLoader(tensor_ds, batch_size=16, shuffle=True)

import torch.nn as nn
m,n = X_tensor.shape
model = nn.Sequential(
    nn.Linear(n, 16),
    nn.ReLU(),    
    nn.Linear(16, 8),
    nn.ReLU(),    
    nn.Linear(8, 1)
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 1000

for i in range(epochs):
    model.train()
    cost= 0

    for x_batch, y_batch in tensor_dl:
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i%100==0:
        cost = loss.item()
        print(f'epoch={i}, cost={cost:.3f}')


# predict
model.eval()

def decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min()-0.5, X[:, 0].max()+0.5
    y_min, y_max = X[:, 1].min()-0.5, X[:, 1].max()+0.5

    # gx(300,300) = 90,000, gy(300,300)
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    X_test = np.c_[gx.ravel(), gy.ravel()]
    X_tensor = torch.tensor(X_test, dtype=torch.float32)

    Z = model(X_tensor)
    A = torch.sigmoid(Z).numpy()

    A = A.reshape(gx.shape)

    plt.contourf(gx, gy, A, cmap='coolwarm', levels=20, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap='coolwarm', s=30)
    plt.show()

with torch.no_grad():
    decision_boundary(model, X, y)


