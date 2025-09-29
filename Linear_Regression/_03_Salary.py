import numpy as np
import pandas as pd

# local machine
import os
curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)
file_path = os.path.join(curr_dir, 'Salary_dataset.csv')
df = pd.read_csv(file_path)

# google colab
# from google.colab import files
# # file_dict -> { '파일명.csv': filedata } 형태로 content에 저장
# file_dict = files.upload()
# file_path = list(file_dict.keys())[0]
# df = pd.read_csv(file_path)

print("First 5 records:", df.head())

import matplotlib.pyplot as plt
plt.scatter(df['YearsExperience'], df['Salary'])
plt.show()

from sklearn.model_selection import train_test_split
X = df['YearsExperience'].to_numpy().astype('float32')
y = df['Salary'].to_numpy().astype(np.float32)

train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2,shuffle=True)
print(train_X.shape, test_X.shape, train_y.shape, test_y.shape)

from torch.utils.data import TensorDataset, DataLoader
import torch

X_tensor = torch.from_numpy(train_X)
y_tensor = torch.from_numpy(train_y)
train_ds = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

X_tensor = torch.from_numpy(test_X)
y_tensor = torch.from_numpy(test_y)
test_ds = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_ds, batch_size=32)

import torch.nn as nn

# Sequential 모델 정의
model = nn.Sequential(
    nn.Linear(1, 1)
)

# 손실 함수와 옵티마이저 
criterion = nn.MSELoss()
# model.parameters()는 w, b를 생성반환
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# 데이터 차원 확장 (입력 shape을 [N, 1]로 바꿔야 함)
def add_dim(x):
    return x.view(-1, 1).float()

# 학습 루프
epochs = 1000
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch = add_dim(x_batch)
        y_batch = add_dim(y_batch)

        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # w, b update

        total_loss += loss.item()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 테스트
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = add_dim(x_batch)
        pred = model(x_batch)

        all_preds.append(pred.view(-1).numpy())
        all_targets.append(y_batch.numpy())

# numpy로 변환해서 flatten
y_pred = np.concatenate(all_preds) 
y_true = np.concatenate(all_targets)

from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f'R2 score: {r2:.4f}')

# 시각화

# 모델 파라미터 가져오기
weight = model[0].weight.item()
bias = model[0].bias.item()

# X축 범위 설정
x_line = np.linspace(test_X.min(), test_X.max(), 100)
y_line = weight * x_line + bias

plt.figure(figsize=(8, 5))
plt.scatter(test_X, test_y, color='red', label='Actual')
plt.scatter(test_X, y_pred, color='blue', label='Predicted')
plt.plot(x_line, y_line, color='blue', label='Regression Line') 
plt.legend()
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Prediction")
plt.show()