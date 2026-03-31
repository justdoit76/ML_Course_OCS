import numpy as np

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def compute_cost(X, y, W, b, _lambda=0):
    m = X.shape[0]
    cost = 0

    for i in range(m):
        z = X[i]*W + b
        y_hat = sigmoid(z)
        cost += -y[i]*np.log(y_hat)-(1-y[i])*np.log(1-y_hat)

    L2 = _lambda/(2*m)*(W**2)
    cost = cost / m + L2
    return cost

def compute_gradient(X, y, W, b, _lambda=0):
    m = X.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        z = X[i]*W+b
        y_hat = sigmoid(z)        

        dj_dw += (y_hat-y[i])*X[i]
        dj_db += (y_hat-y[i])

    L2 = _lambda/m*W
    dj_dw = dj_dw / m + L2
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(X, y, W, b, lr, epochs, _lambda=0):
    for i in range(epochs):
        cost = compute_cost(X, y, W, b, _lambda)

        dj_dw, dj_db = compute_gradient(X, y, W, b, _lambda)

        W = W - lr * dj_dw
        b = b - lr * dj_db

        if i%100==0:
            print(f'epoch={i}, W={W:.3f}, b={b:.3f}, cost={cost:.3f}')

    return W, b
