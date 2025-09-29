import numpy as np
import pandas as pd
import os

curr_path = os.path.abspath(__file__)
curr_dir  = os.path.dirname(curr_path)
file_path = os.path.join(curr_dir, 'Student_Performance.csv')

df = pd.read_csv(file_path)
print(df.head())

# 1. dataset
from sklearn.model_selection import train_test_split

y_label = df.columns[-1]

# to_numpy()는 안해도 sklean 내부에서 변환, 하지만 사용추천
# get_dummies : Yes/No -> 1:0

df = pd.get_dummies(df)

X = df.drop(y_label, axis=1).to_numpy().astype('float32')
y = df[y_label].to_numpy().astype('float32')

X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# model
from torch.utils.data import TensorDataset, DataLoader
import torch

X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)
y_tensor = y_tensor.view(-1, 1)
train_ds = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

X_tensor = torch.from_numpy(X_test)
y_tensor = torch.from_numpy(y_test)
y_tensor = y_tensor.unsqueeze(1)
test_ds = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_ds, batch_size=8)

import torch.nn as nn

print(X.shape)
model = nn.Sequential( 
    nn.Linear(X.shape[1], 1),
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam( model.parameters(), lr=0.002 )
#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# learn
epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch+1) % 10 == 0:
        print(f'{epoch+1} : {total_loss:.4f}')

# test
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        pred = model(x_batch)  # (batch_size, 1)
        all_preds.append(pred.view(-1).cpu().numpy())   # flatten해서 numpy
        all_targets.append(y_batch.view(-1).cpu().numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_targets)

from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f'R2 score: {r2:.4f}')