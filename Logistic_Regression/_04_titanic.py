import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("titanic")

print(df.head(5))
# | 컬럼명            | 의미                                                     
# | ---------------- | ------------------------------------------------------
# | **survived**     | 생존 여부 (0 = 사망, 1 = 생존)
# | **pclass**       | 선실 등급 (1 = 1등석, 2 = 2등석, 3 = 3등석)
# | **sex**          | 성별 (male, female)
# | **age**          | 나이 (년 단위, 결측치 있음)
# | **sibsp**        | 함께 탑승한 형제자매/배우자 수
# | **parch**        | 함께 탑승한 부모/자녀 수
# | **fare**         | 탑승 요금 (화폐 단위)
# | **embarked**     | 탑승 항구 (C = Cherbourg, Q = Queenstown, S = Southampton)
# | **class**        | 선실 등급 문자형 (First, Second, Third)
# | **who**          | 탑승자 유형 (man, woman, child)
# | **adult\_male**  | 성인 남성 여부 (True/False)
# | **deck**         | 객실 데크 위치 (A\~G, 결측치 있음)
# | **embark\_town** | 탑승 도시 (Cherbourg, Queenstown, Southampton)
# | **alive**        | 생존 여부 문자형 (yes, no)
# | **alone**        | 혼자인지 여부 (True = 혼자, False = 가족 있음)


# 생존자별 성별
sns.countplot(data=df, x='survived', hue='sex')
plt.title('Survival by Sex')
plt.show()

# 나이 분포 비교 (생존자 vs 사망자)
sns.histplot(data=df, x='age', hue='survived', multiple='stack', kde=True)
plt.title('Age Distribution by Survival')
plt.show()

# 객실 등급별 생존률
sns.barplot(data=df, x='class', y='survived')
plt.title('Survival Rate by Passenger Class')
plt.show()

# embark_town, alive 제외
df = df.drop(columns=['embark_town', 'alive'])
print()
print(df.head(10))


# 수치, 이진, 다중범주형 분리
num_cols = ['pclass', 'age', 'sibsp', 'parch', 'fare']
bin_cat_cols = ['sex', 'adult_male', 'alone']
mul_cat_cols = ['embarked', 'class', 'who', 'deck']

from sklearn.preprocessing import StandardScaler, LabelEncoder

# 결측치를 확인하고 변환

# 1.수치형
print(df[num_cols].isnull().sum())
df['age'] = df['age'].fillna(df['age'].median())

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 2.이진 범주형
print(df[bin_cat_cols].isnull().sum())

le = LabelEncoder()
for col in bin_cat_cols:
    df[col] = le.fit_transform(df[col])

# 3.다중 범주형
print(df[mul_cat_cols].isnull().sum())

# 가장 흔한 값으로 채움 (예: 'S')
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# deck 결측치가 많지만 컬럼 삭제 X, 결측자체(객실을 배정못받은 승객)가 생존율과 연관가능성.
# 일단 Unknown 카테고리추가
df['deck'] = df['deck'].cat.add_categories('Unknown')
# 이후 NaN을 Unknown으로
df['deck'] = df['deck'].fillna('Unknown')
df = pd.get_dummies(df, columns=mul_cat_cols, drop_first=True)

print()
print(df.head(10))


# Data set
X = df.iloc[:, 1:].to_numpy().astype('float32')
y = df['survived'].to_numpy().reshape(-1, 1).astype('float32')

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


# model
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(X.shape[1], 1),        
    # 출력층에 sigmoid를 쓰지 않은 이유는 BCEWithLogitsLoss가 이미 포함
)

y_count = df['survived'].value_counts()
pos_weight = torch.tensor([y_count[0] / y_count[1]])

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

J_train = []
J_cv = []

all_preds = []
all_targets = []

epochs = 500

for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0

    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    # 평균 train loss
    epoch_train_loss = total_train_loss / len(train_dl)
    J_train.append(epoch_train_loss)

    # ===> 여기서 validation loss 계산 (Jcv) <===

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for x_val, y_val in test_dl:
            val_pred = model(x_val)
            val_loss = criterion(val_pred, y_val)
            total_val_loss += val_loss.item()

            probs = torch.sigmoid(val_pred)
            preds = (probs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_val.cpu().numpy())

    epoch_val_loss = total_val_loss / len(test_dl)
    J_cv.append(epoch_val_loss)

    # TensorBoard 기록
    writer.add_scalar("Loss/Train", epoch_train_loss, epoch)
    writer.add_scalar("Loss/Validation", epoch_val_loss, epoch)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} : J_train = {epoch_train_loss:.4f}, J_cv = {epoch_val_loss:.4f}")

writer.close()


from sklearn.metrics import classification_report
print(classification_report(all_targets, all_preds, digits=4))