import numpy as np
import os

# open csv
curr_path = os.path.abspath(__file__)
curr_dir  = os.path.dirname(curr_path)
file_path = os.path.join(curr_dir, 'diabetes.csv')

import pandas as pd
df = pd.read_csv(file_path)
print(df.head(5))

X = df.drop( df.columns[-1], axis=1).to_numpy().astype('float32')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

y_label = df.columns[-1]
y = df[y_label].to_numpy().reshape(-1,1).astype('float32')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

import torch
from torch.utils.data import DataLoader, TensorDataset
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)
train_ds = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

X_tensor = torch.from_numpy(X_test)
y_tensor = torch.from_numpy(y_test)
test_ds = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_ds, batch_size=32)

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(X.shape[1], 1),
    # model이 복잡하면 overfit
    # nn.ReLU(),
    # nn.Linear(32, 16),    
    # nn.ReLU(),
    # nn.Linear(16, 1),
)


# PyTorch의 BCEWithLogitsLoss는 양성 클래스에만 가중치를 줄 수 있는 pos_weight 인자를 지원
# Pima 데이터셋은 클래스 0과 1의 비율이 불균형 (약 65:35)
# y_dataset 중 0이 500, 1이 268이면 500/268=1.87 
# 즉 양성클래스 1의 손실을 1.87배 크게 계산
y_count = df[y_label].value_counts()
print(y_count)
pos_weight = torch.tensor([y_count[0] / y_count[1]])

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

epochs = 500

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

model.eval()

from sklearn.metrics import classification_report

all_preds = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

print(classification_report(all_targets, all_preds, digits=4))
