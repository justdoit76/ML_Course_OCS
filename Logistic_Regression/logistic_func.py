import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))


def compute_cost(X, y, W, b, Lambda=0):
    m, n = X.shape

    z = np.dot(X, W) + b
    y_hat = sigmoid(z)

    # epsilon, if y_hat==0, np.log(0)=-∞ 방지
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1-epsilon)

    loss = -y*np.log(y_hat) - (1-y)*np.log(1-y_hat)
    regular = Lambda/(2*m)*np.sum(W**2)
    cost = 1/m*np.sum(loss) + regular
    return cost

def compute_gradient(X, y, W, b, Lambda=0):
    """
    X (m, n)
    y (m, 1)
    W (n, 1)
    z (m, 1)
    y_hat (m, 1)
    err (m, 1)
    """
    m, n = X.shape

    z = np.dot(X, W) + b
    y_hat = sigmoid(z)
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
            print(f'epoch={i}, W={np.round(W,3)}, b={b:.3f}, cost={cost:.3f}')

    return W, b


def confusion_matrix(y_true, y_pred):
    cm = {'TP':0, 'TN':0, 'FP':0, 'FN':0}
    cnt = len(y_true)

    for i in range(cnt):
        if y_true[i] and y_pred[i]:
            cm['TP']+=1
        elif not y_true[i] and not y_pred[i]:
            cm['TN']+=1
        elif not y_true[i] and y_pred[i]:
            cm['FP']+=1
        elif y_true[i] and not y_pred[i]:
            cm['FN']+=1

    return cm

    

