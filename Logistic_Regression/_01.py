import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy

# load the dataset
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'lung_cancer_data.csv')

df = pd.read_csv(csv_path)
 
np_array = df.to_numpy()
print(np_array.shape)
 
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Scatter plot between specific features', fontsize=16)
 
sns.scatterplot(x='Age', y='SmokingDuration', hue='LungCancer', data=df, ax=axes[0], alpha=0.7)
sns.scatterplot(x='SmokingDuration', y='SmokingAmount', hue='LungCancer', data=df, ax=axes[1], alpha=0.7)
 
plt.show()
 
 
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g
 
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0
 
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
 
        cost += -y[i] * np.log(f_wb_i) - (1- y[i]) * np.log(1-f_wb_i)
    cost /= m
    return cost
 
def compute_gradient_logistic(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0
 
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        err_i = f_wb_i - y[i]
 
        for j in range(n):
            dj_dw[j] += err_i * X[i][j]
        dj_db += err_i
 
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db
 
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
 
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
 
        cost = compute_cost_logistic(X, y, w, b)
        J_history.append(cost)
 
        if i%100 == 0:
            print(f'Iteration {i}: w = {w}, b = {b:.3f}, cost = {J_history[-1]:.3f}')
 
    return w, b, J_history
 
# init parameters
w = np.zeros_like(np_array[0, :-1])
b = 0
alpha = 0.001
num_iters = 1000
 
# machine learning
w_final, b_final, J_history = gradient_descent(np_array[:, :-1], np_array[:, -1], w, b, alpha, num_iters)
print(f'Final w: {w_final}, Final b: {b_final}')
 
# cost function convergence plot
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.show()
 
# prediction
def predict(w, b, x):
    z = np.dot(x, w) + b
    g = sigmoid(z)
    if g >= 0.5:
        return 1
    else:
        return 0
 
accuracy = 0
sample_cnt = 30
     
for x in np_array[:sample_cnt, :]:
    x_test = x[:-1]
    y_test = int(x[-1])
    y_pred = predict(w_final, b_final, x_test)
    accuracy += (y_pred == y_test)
    print(f'Input {x_test} :\tPred:{y_pred} True:{y_test}')
 
print(f'Accuracy: {accuracy/sample_cnt:.2%}')
