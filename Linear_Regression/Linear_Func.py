import numpy as np

def compute_cost(X, y, W, b, Lambda=0):
    m, n = X.shape
    y_hat = np.dot(X, W) + b
    err = np.sum( (y_hat-y)**2 )

    cost = 1/(2*m)*err
    regular = Lambda/(2*m)*np.sum(W**2)
    total_cost = cost + regular

    return total_cost

def compute_gradient(X, y, W, b, Lambda=0):
    m, n = X.shape
    y_hat = np.dot(X, W) + b
    err = y_hat - y    

    dj_dw = np.dot(X.T, err) / m
    dj_db = np.sum(err) / m

    regular = Lambda/m*W
    dj_dw += regular
    return dj_dw, dj_db

def gradient_descent(X, y, W, b, lr, epochs, Lambda=0):
    for i in range(epochs):
        dj_dw, dj_db = compute_gradient(X, y, W, b, Lambda)

        W = W - lr * dj_dw
        b = b - lr * dj_db

        if i%100==0:
            cost = compute_cost(X, y, W, b, Lambda)
            print(f'epoch={i}, W={np.round(W, 3)}, b={b:.3f}, cost={cost:.3f}')

    return W, b