from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

C = 5
X, y = make_blobs(n_samples=500, centers=5, cluster_std=0.8, random_state=10)

X = X.astype(np.float32)
y = y.astype(np.longlong)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10')
plt.show()

import torch
from torch.utils.data import TensorDataset, DataLoader

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

tensor_ds = TensorDataset(X_tensor, y_tensor)
tensor_dl = DataLoader(tensor_ds, batch_size=64, shuffle=True)

import torch.nn as nn
m, n = X_tensor.shape
model = nn.Sequential(
    nn.Linear(n, C)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam( model.parameters(), lr=0.01 )
epochs = 500

for i in range(epochs):
    model.train()
    cost = 0

    for x_batch, y_batch in tensor_dl:
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss.item()

    cost /= len(tensor_dl)

    if i%100==0:
        print(f'epoch={i}, cost={cost:.3f}')

# predict
model.eval()
with torch.no_grad():
    from nn_plot_func import plot_decision_boundary, plot_hidden_layer_with_neurons

    X_test = np.array([
        [0, -5]
    ])
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    Z = model(X_test_tensor)
    A = torch.softmax(Z, dim=1)
    M = torch.argmax(A, dim=1)

    print(f'Z={Z}')
    print(f'A={A}')
    print(f'M={M}')

    plot_decision_boundary(model, X, y, C)      