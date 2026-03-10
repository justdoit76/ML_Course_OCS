import numpy as np
import pandas as pd
import os

curr_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_path)
file_path = os.path.join(curr_dir, 'Student_Performance.csv')

df = pd.read_csv(file_path)
#print(df.head(5))

y_label = df.columns[-1]
X = df.drop(y_label, axis=1)
X = pd.get_dummies(X, drop_first=True).to_numpy().astype(np.float32)
y = df[y_label].to_numpy().astype(np.float32)
#print(X)
#print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Z-score
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_p = ss.fit_transform(X_train)

m, n = X_train_p.shape

# init hyper parameters
W = np.zeros(n)
b = 0
lr = 0.002
epochs = 3000
Lambda = 0

from Linear_Func import gradient_descent
W_final, b_final = gradient_descent(X_train_p, y_train, W, b, lr, epochs, Lambda)

def predict(X, W, b):
    return np.dot(X, W) + b

X_test_p = ss.transform(X_test)
y_pred = predict(X_test_p, W_final, b_final)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f'r2={r2:.3f}')