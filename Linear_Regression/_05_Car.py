import os
import numpy as np
import pandas as pd

# dataset
curr_path = os.path.abspath(__file__)
curr_dir  = os.path.dirname(curr_path)
file_path = os.path.join(curr_dir, 'Car_dataset.csv')

df = pd.read_csv(file_path)

# 데이터 셋 전처리 필요

# 1. remove 'name' column
df = df.drop('name', axis=1)
print(df.columns)

# 2. categorical(범주형), numberic(수치형)
num_cols = ['year', 'selling_price', 'km_driven']
cat_cols = ['fuel', 'seller_type', 'transmission', 'owner']

# 3. one hot encoding(범주형 : fuel, seller type, transmission, owner)
df_cat = pd.get_dummies(df[cat_cols])

# 4. numberic 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_num = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)

# 5. df_num + df_cat
df_final = pd.concat([df_num, df_cat], axis=1)
X = df_final.to_numpy().astype('float32')

y_label = 'selling_price'
#y = df[y_label].to_numpy().astype('float32')
# log1p = log(x+1) = x가 10 인경우 11이 되어 자연로그 loge(11), 약 2.3978...
y = np.log1p(df[y_label].to_numpy()).astype('float32').reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
print(X_train.shape)

# tensor dataset
from torch.utils.data import DataLoader, TensorDataset
import torch

X_tensor = torch.from_numpy(X_train)
y_tensor = torch.from_numpy(y_train)
train_ds = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

X_tensor = torch.from_numpy(X_test)
y_tensor = torch.from_numpy(y_test)
test_ds = TensorDataset(X_tensor, y_tensor)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)

# model
import torch.nn as nn
model = nn.Sequential( 
    nn.Linear(X.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 1),    
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

# inverse of log1p
y_pred = np.expm1(y_pred)
y_true = np.expm1(y_true)

from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(f'R2 score: {r2:.4f}')