from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt


C = 8
X, y =  make_blobs(n_samples=500, centers=C, cluster_std=0.6)
X = X.astype(np.float32)

plt.scatter(X[:,0], X[:, 1], c=y, cmap='tab10')
plt.colorbar(label='Class', ticks=range(C))
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
    nn.Linear(n, C*2),
    nn.ReLU(),      
    nn.Linear(C*2, C),
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 1000

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

    if i%100==0:        
        cost = cost / len(tensor_dl)
        print(f'epoch={i}, cost={cost:.3f}')


# predict
model.eval()

from plot_nn_func import plot_decision_boundary, plot_hidden_layer_with_neurons
plot_decision_boundary(model, X, y, C)
plot_hidden_layer_with_neurons(model, X, y, 1, C*2)
plot_hidden_layer_with_neurons(model, X, y, 2, C*2)
plot_hidden_layer_with_neurons(model, X, y, 3, C)