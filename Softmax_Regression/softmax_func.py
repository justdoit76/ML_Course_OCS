import numpy as np

def softmax(Z):
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp_z = np.exp(Z)
    return exp_z  / np.sum(exp_z, axis=1, keepdims=True)

def compute_cost(X, y, W, b, Lambda = 0):
    m, n = X.shape
    Z = np.dot(X, W) + b
    A = softmax(Z)

    cost = -np.sum(y*np.log(A)) / m
    regular = Lambda / (2*m) * np.sum(W**2)

    cost += regular
    return cost

def compute_gradient(X, y, W, b, Lambda=0):
    m, n = X.shape

    Z = np.dot(X, W) + b
    A = softmax(Z)

    err = A-y
    dj_dw = np.dot(X.T, err) / m
    dj_db = np.sum(err, axis=0) / m

    regular = Lambda / m * W
    dj_dw += regular

    return dj_dw, dj_db

def gradient_descent(X, y, W, b, lr, epochs, Lambda=0):
    for i in range(epochs):
        dj_dw, dj_db = compute_gradient(X, y, W, b, Lambda)

        W = W - lr * dj_dw
        b = b - lr * dj_db

        if i%100==0:
            cost = compute_cost(X, y, W, b, Lambda)
            print(f'epochs={i}, cost={cost:.3f}')

    return W, b