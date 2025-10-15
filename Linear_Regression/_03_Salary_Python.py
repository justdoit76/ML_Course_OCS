import numpy as np
import matplotlib.pyplot as plt

import os
path = os.path.abspath(__file__)
dir = os.path.dirname(path)
file = os.path.join(dir, 'Salary_dataset.csv')

import pandas as pd
df = pd.read_csv(file)
print(df.head(5))
keys = df.keys()

X = df[keys[1]].to_numpy()
y = df[keys[2]].to_numpy()
print(X.shape, y.shape)

plt.scatter(X, y)
plt.show()

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w*X[i]+b
        cost += (f_wb-y[i])**2

    total_cost = 1/(2*m)*cost
    return total_cost

def compute_gradient(X, y, w, b):
    m = X.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w*X[i]+b
        dj_dw += (f_wb-y[i])*X[i]
        dj_db += f_wb-y[i]

    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, num_iters):
    J_history = []
    p_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        J_history.append(compute_cost(X,y,w,b))
        p_history.append((w, b))

        if i%10==0:
            print(f'Iter:{i}, w={w:.3f}, b={b:.3f}, cost={J_history[-1]:.3f}')

    return w, b, J_history, p_history


# init parameters
w = 0
b = 0
alpha = 0.001
num_iters = 10000

# machine learning
w_final, b_final, J_hist, p_hist = gradient_descent(X, y, w, b, alpha, num_iters)
print(f'Final w={w_final:.3f}, b={b_final:.3f}')


# predict
def predict(w, b, x):
    return w*x+b

X_test = np.array([1.3, 1.4, 2.5, 5.5, 7, 7.5,10])
y_test = []
for x in X_test:
    y_pred = predict(w_final, b_final, x)
    y_test.append(y_pred)
    print(f'x={x:.1f}, y={y_pred:.1f}')     

plt.scatter(X, y)
plt.plot(X_test, y_test, color='red')
plt.show()