import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)
file_path = os.path.join(curr_dir, 'lung_cancer_data.csv')

df = pd.read_csv(file_path)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Scatter plot between specific features', fontsize=16)
 
sns.scatterplot(x='Age', y='SmokingDuration', hue='LungCancer', data=df, ax=axes[0], alpha=0.7)
sns.scatterplot(x='SmokingDuration', y='SmokingAmount', hue='LungCancer', data=df, ax=axes[1], alpha=0.7)
 
plt.show()

X = df.iloc[:, :-1].to_numpy().astype(np.float32)
y = df.iloc[:, -1].to_numpy().astype(np.float32)
# print(X)
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_p = ss.fit_transform(X_train)

m, n = X_train_p.shape
W = np.zeros(n)
b = 0
lr = 0.01
epochs = 2000
Lambda = 0

from logistic_func import gradient_descent, sigmoid
W_final, b_final = gradient_descent(X_train_p, y_train, W, b, lr, epochs, Lambda)

def predict(X, W, b, threshold=0.4):
    Z = np.dot(X, W) + b
    A = sigmoid(Z)
    y_pred = (A>=threshold).astype(int)
    return y_pred

X_test_p = ss.transform(X_test)
y_pred = predict(X_test_p, W_final, b_final)

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
a = accuracy_score(y_test, y_pred)
p = precision_score(y_test, y_pred)
r = recall_score(y_test, y_pred)

print(f'accuracy={a:.3f}')
print(f'precisiton={p:.3f}')
print(f'recall={r:.3f}')

cr = classification_report(y_test, y_pred)
print(cr)

