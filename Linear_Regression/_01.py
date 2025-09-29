import numpy as np
import matplotlib.pyplot as plt
import copy
 
X = np.array([1, 5, 10])
y = np.array([10, 50, 100])
 
m = X.shape[0]
print('Number of datasets:', m)
 
plt.xlabel('Learning time')
plt.ylabel('Score')
plt.plot(X, y)
plt.scatter(X, y, color='red')
plt.show()
 
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * X[i] + b
        cost += (f_wb - y[i]) ** 2
 
    total_cost = 1 / (2*m) * cost
    return total_cost
 
def compute_gradient(X, y, w, b):
    m = X.shape[0]
    dj_dw = 0
    dj_db = 0
 
    for i in range(m):
        f_wb = w * X[i] + b
        dj_dw += (f_wb - y[i]) * X[i]
        dj_db += f_wb - y[i] 
 
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db
 
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    w = copy.deepcopy(w_in)
    b = b_in
 
    J_history = []
    p_history = []
 
    for i in range(num_iters):        
        dj_dw, dj_db = compute_gradient(X, y, w, b)
 
        w = w - alpha * dj_dw
        b = b - alpha * dj_db 
      
        J_history.append( compute_cost(X, y, w, b) )
        p_history.append( (w, b) )
 
        if i%10 == 0:
            print(f'iter {i}: w = {w:.3f}, b = {b:.3f}, cost = {J_history[-1]:.3f}')
 
    return w, b, J_history, p_history
 
# init parameters
w = 0
b = 0
alpha = 0.001
num_iters = 1000
 
# machine learning
w_final, b_final, J_history, p_history = gradient_descent(X, y, w, b, alpha, num_iters)
print(f'Final w: {w_final}, Final b: {b_final}')
 
# prediction
def predict(w, b, x):
    return w * x + b
 
X_test = np.array([2, 4, 6, 8])
for x in X_test:
    y_pred = predict(w_final, b_final, x)
    print(f'Predicted score for {x} hours of learning: {y_pred:.2f}')