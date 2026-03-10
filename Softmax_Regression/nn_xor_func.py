import numpy as np

def sigmoid(Z):
    return 1/(1+ np.exp(-Z))

def compute_cost(X, y, W, b, Lambda=0):
    m, n = X.shape
    Z = np.dot(X, W) + b
    A = sigmoid(Z)

    epsilon = 1e-15
    A = np.clip(A, epsilon, 1-epsilon)
    
    L = -y*np.log(A)-(1-y)*np.log(1-A)
    cost = np.sum(L) / m

    regular = Lambda/(2*m)*np.sum(W**2)
    cost += regular
    return cost

def compute_gradient(X, y, W, b, Lambda=0):
    m, n = X.shape
    Z = np.dot(X, W) + b
    A = sigmoid(Z)

    err = A-y

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
            print(f'epochs={i}, W={np.round(W, 3)}, b={b:.3f}, cost={cost:.3f}')

    return W, b


def mlp_forward(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    
    cache = (Z1, A1, Z2, A2)
    return A2, cache

def mlp_cost(y, A):
    epsilon = 1e-15
    A = np.clip(A, epsilon, 1-epsilon)
    
    loss =  -np.mean(y * np.log(A) + (1 - y) * np.log(1 - A))
    return loss

def mlp_backprop(X, y, cache, W2):   
    """
    순전파 과정
    Z1 = X·W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1·W2 + b2
    A2 = sigmoid(Z2)

    """
    Z1, A1, Z2, A2 = cache
    m, n = X.shape
    # 1. 출력층 오차 계산 (출력결과, 실제값 차이)
    # 1.1 dL/dA2
    dL_dA2 = -(y / A2) + ((1 - y) / (1 - A2))

    # 1.2 dA2/dZ2
    dA_dZ2 = A2 * (1 - A2)

    # 1.3 체인룰
    dZ2 = dL_dA2 * dA_dZ2
    # dZ2 = A2-y (1~3, 한줄로 가능)

    # 2. 출력층 가중치 업데이트(dZ2는 W2, b2에 의해 영향, 책임묻기)    
    dW2 = np.dot(A1.T , dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # 3. 은닉층(W1, b1) 오차 전파
    # 3.1 출력층의 오차(dZ2)를 가중치 W2를 타고 역방향으로 보냄
    dA1 = np.dot(dZ2, W2.T)
    # 3.2 시그모이드 미분    
    dZ1 = dA1 * A1 * (1 - A1)

    # 4. 입력층 가중치 업데이트(dZ1은 W1, b1에 의해 영향, 책임묻기)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2

def mlp_training(X, y, W1, b1, W2, b2, lr, epochs):
    for i in range(epochs):
        A, cache = mlp_forward(X, W1, b1, W2, b2)
        cost = mlp_cost(y, A)

        dW1, db1, dW2, db2 = mlp_backprop(X, y, cache, W2)

        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2

        if i%100==0:
            print(f'epoch={i} cost{cost:.3f}')

    return W1, b1, W2, b2