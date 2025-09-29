import numpy as np
import matplotlib.pyplot as plt
import copy
 
np.random.seed(0)
X = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([10, 20, 30, 40, 50, 60, 370])
 
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

# L2 정규화 (Ridge Regression)
def compute_cost_regular(X, y, w, b, lambda_):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * X[i] + b
        cost += (f_wb - y[i]) ** 2

    reg_cost = (lambda_ / (2*m)) * (w**2)
    total_cost = 1 / (2*m) * cost + reg_cost
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

# L2 정규화 (Ridge Regression)
def compute_gradient_regular(X, y, w, b, lambda_):
    m = X.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * X[i] + b
        dj_dw += (f_wb - y[i]) * X[i]
        dj_db += f_wb - y[i]

    dj_dw /= m
    dj_db /= m

    dj_dw += (lambda_ / m) * w

    return dj_dw, dj_db

 
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_ = 0.01, regular = False):
    w = copy.deepcopy(w_in)
    b = b_in
 
    J_history = []
    p_hiistory = []

    cost = 0
 
    for i in range(num_iters):
        #dj_dw, dj_db = compute_gradient(X, y, w, b)
        if regular:
            dj_dw, dj_db = compute_gradient_regular(X, y, w, b, lambda_)
            cost = compute_cost_regular(X, y, w, b, lambda_)
        else:
            dj_dw, dj_db = compute_gradient(X, y, w, b)
            cost = compute_cost(X, y, w, b)
 
        w = w - alpha * dj_dw
        b = b - alpha * dj_db 
        
        J_history.append( cost )
        p_hiistory.append( (w, b) )
 
        # if i%10 == 0:
        #     print(f'Iteration {i}: w = {w:.3f}, b = {b:.3f}, cost = {J_history[-1]:.3f}')
 
    return w, b, J_history, p_hiistory
 
# init parameters
w = 0
b = 0
alpha = 0.01
num_iters = 10000

lambda_ = 50.0
 
# Regularization X
w1, b1, J_history, p_history = gradient_descent(X, y, w, b, alpha, num_iters, lambda_, regular=False)
print(f'Regular(X) w: {w1}, Final b: {b1}')

# Regularization O
w2, b2, J_history, p_history = gradient_descent(X, y, w, b, alpha, num_iters, lambda_, regular=True)
print(f'Regular(O) w: {w2}, Final b: {b2}')
 
# prediction
def predict(w, b, x):
    return w * x + b
 
X_test = np.array([2, 4, 6, 8])
for x in X_test:
    y_pred1 = predict(w1, b1, x)
    y_pred2 = predict(w2, b2, x)
    print(f'Predicted score for {x} hours of learning (Regular X): {y_pred1:.2f}')
    print(f'Predicted score for {x} hours of learning (Regular O): {y_pred2:.2f}')


plt.plot(X_test, predict(w1, b1, X_test), label='No Regularization', color='orange')
plt.plot(X_test, predict(w2, b2, X_test), label=f'With Regularization (λ={lambda_})', color='green')
plt.scatter(X, y, color='red')
plt.title('Overfitting vs Regularized Fit')
plt.legend()
plt.show()