import numpy as np
import pandas as pd
import os

curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)
file_path = os.path.join(curr_dir, 'Iris.csv')

df = pd.read_csv(file_path)
print(df.head(5))

X = df.iloc[:, 1:-1]
y_label = df.columns[-1]

# one hot encoding
from sklearn.preprocessing import StandardScaler, LabelEncoder
le = LabelEncoder()
y_int = le.fit_transform(df[y_label])
C = len(le.classes_)
# 단위벡터
I = np.eye(C)
y_onehot = I[y_int]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, shuffle=True)
m, n = X_train.shape

# Z-score
ss = StandardScaler()
X_train_p = ss.fit_transform(X_train)



#W = np.zeros((n, C))
W = np.random.randn(n, C) * 0.01
print(W)
b = np.zeros(C)
lr = 0.01
epochs = 3000
Lambda = 0

from softmax_func import gradient_descent, softmax
W_final , b_final = gradient_descent(X_train_p, y_train, W, b, lr, epochs, Lambda)


# predict
def predict(X, W, b):
    Z = np.dot(X, W) + b
    A = softmax(Z)
    y_pred = np.argmax(A, axis=1)
    return y_pred

X_test_p = ss.transform(X_test)
y_pred = predict(X_test_p, W_final, b_final)
y_true = np.argmax(y_test, axis=1)

print(y_true)
print(y_pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

a = accuracy_score(y_true, y_pred)
p = precision_score(y_true, y_pred, average='macro')
r = recall_score(y_true, y_pred, average='macro')

print(f'accuracy={a:.3f}')
print(f'precision={p:.3f}')
print(f'recall={r:.3f}')

cr = classification_report(y_true, y_pred)
print(cr)