import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# import dataset
X, y = make_moons(n_samples=500, noise=0.2, random_state=10)

X = X.astype(np.float32)
y = y.reshape(-1, 1).astype(np.float32)

plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap='coolwarm')
plt.show()

# numpy -> tensor
import torch
from torch.utils.data import TensorDataset, DataLoader

X_tensor = torch.from_numpy(X)
y_tensor = torch.tensor(y)

tensor_ds = TensorDataset(X_tensor, y_tensor)
tesnor_dl = DataLoader(tensor_ds, batch_size=16, shuffle=True)

# model
import torch.nn as nn

m, n = X_tensor.shape

model = nn.Sequential(
    nn.Linear(n, 16),
    nn.ReLU(),
    nn.Linear(16,1)   
)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 1000

for i in range(epochs):
    model.train()
    cost = 0

    for x_batch, y_batch in tesnor_dl:
        # forward propagration
        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)

        # backprop.
        optimizer.zero_grad()
        loss.backward()
        # update w, b
        optimizer.step()

    if i%100==0:
        cost = loss.item()
        print(f'epoch={i}, cost={cost:.3f}')
    
# predict
model.eval()

def decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    X_test = np.c_[gx.ravel(), gy.ravel()]
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    Z = model(X_test_tensor)
    A = torch.sigmoid(Z).numpy()

    A = A.reshape(gx.shape)

    plt.contourf(gx, gy, A, levels=20, cmap='coolwarm', alpha=0.5)
    plt.scatter(X[:, 0], X[:,  1], c=y[:, 0], cmap='coolwarm')
    plt.show()

with torch.no_grad():
    decision_boundary(model, X, y)