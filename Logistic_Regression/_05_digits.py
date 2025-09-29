import sklearn
import numpy as np

# 1. Dataset
digits = sklearn.datasets.load_digits()

# digits 데이터는 모든 특성이 이미 수치형이고, 결측치도 없고,
# 라벨도 숫자(int)로 구성되어있으므로 Dataframe 처리 불필요
X = digits.data.astype(np.float32)
y = digits.target.astype(np.int64)
print(X[0], X.shape)
print(y[0], y.shape)

import matplotlib.pyplot as plt

# single image
# plt.imshow(digits.images[0], cmap='grey')
# plt.title(f'Label: {digits.target[0]}')
# plt.show()

# 몇 개를 시각화할지 설정
num = 20
col = 5
row = (num + col - 1) // col

plt.figure(figsize=(10, 6))

for i in range(num):
    plt.subplot(row, col, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# 2. Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Train, Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

import torch
from torch.utils.data import DataLoader, TensorDataset
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)
train_ds = TensorDataset(X_tensor, y_tensor)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

X_tensor = torch.from_numpy(X_test)
y_tensor = torch.from_numpy(y_test)
test_ds  = TensorDataset(X_tensor, y_tensor)
test_dl  = DataLoader(test_ds, batch_size=32, shuffle=True)

# 4. model
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(X.shape[1], 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 5. learn
epochs = 200

for epoch in range(epochs):
    model.train()
    train_loss = 0

    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if (epoch+1)%10==0:
        print(f'{epoch+1} : {train_loss:.4f}')


# 6. validation
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_dl:
        logits = model(x_batch)
        # argmax    : 최대값의 인덱스 반환
        # max       : 최대값과 인덱스를 모두 반환       
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

# 7. report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(all_targets, all_preds, digits=4))

cm = confusion_matrix(all_targets, all_preds)
print(cm)