import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_cost(X, y, W, b, Lambda = 0):
    m, n = X.shape

    z = np.dot(X, W) + b
    y_hat = sigmoid(z)

    # log(0)==-∞ 대비, y_hat==0 이면 ε, y_hat==1이면 1-ε
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    cost = -y*np.log(y_hat) - (1-y) * np.log(1-y_hat)

    regular = (Lambda/(2*m)) * np.sum(W**2)
    total_cost = np.sum(cost)/m + regular
    return total_cost
    
def compute_gradient(X, y, W, b, Lambda=0):
    m, n = X.shape

    z = np.dot(X, W) + b
    y_hat = sigmoid(z)

    err = y_hat-y
    dj_dw = np.dot(err, X)
    dj_db = np.sum(err)

    dj_dw /= m
    dj_db /= m

    regular = Lambda/m*W
    dj_dw += regular
    return dj_dw, dj_db

def gradient_descent(X, y, W, b, lr, epoch, Lambda=0):
    for i in range(epoch):
        dj_dw, dj_db = compute_gradient(X, y, W, b, Lambda)
        cost = compute_cost(X, y, W, b, Lambda)

        W = W - lr * dj_dw
        b = b - lr * dj_db

        if i%100==0:
            print(f'epoch={i}, W={np.round(W, 3)}, b={b:.3f}, cost={cost:.3f}')

    return W, b
