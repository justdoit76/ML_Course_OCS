import os
curr_path = os.path.abspath(__file__)
curr_dir  = os.path.dirname(curr_path)
file_path = os.path.join(curr_dir, 'iris.csv')

import pandas as pd
df = pd.read_csv(file_path)
print(df.head(5))

num_cols = df.iloc[:, 0:4].columns
cat_cols = df.columns[-1]

X = df[num_cols].to_numpy().astype('float32')

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[cat_cols]).astype('int64')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

import torch
from torch.utils.data import DataLoader, TensorDataset
X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)
train_ds = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_ds, batch_size=16, shuffle= True)

X_tensor = torch.from_numpy(X_test)
y_tensor = torch.from_numpy(y_test)
test_ds = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_ds, batch_size=16, shuffle= True)

import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
    # 출력층을 softmax를 쓰지 않은 이유:
    # nn.CrossEntropyLoss가 내부적으로 
    # softmax, Negative Log Likelihood Loss (NLLLoss, 로그비용함수) 를 함께 수행           
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 1000

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
total_correct = 0
total_samples = 0

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        # torch.max는 (최대값, 인덱스) 리턴
        # ex) utputs = [[2.1, 0.5, 1.3], [0.1, 4.0, 0.9]] preds = [0, 1]
        # 각 행에서 가장 큰 값의 인덱스
        _, preds = torch.max(outputs, dim=1)

        # preds와 같은 현재 y_batch의 합 개수
        total_correct += (preds == y_batch).sum().item()
        # 0은 행을 의미
        total_samples += y_batch.size(0)

test_acc = total_correct / total_samples
print(f'Test Accuracy: {test_acc:.4f}')