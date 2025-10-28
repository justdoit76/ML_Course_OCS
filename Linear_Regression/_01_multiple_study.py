import numpy as np
import matplotlib.pyplot as plt
import copy

# 입력 데이터 (5개의 특성)
# [공부시간, 수면시간, 출석률, 모의고사횟수, 스트레스]
X = np.array([
    [1, 6, 0.7, 1, 8],
    [3, 7, 0.8, 2, 6],
    [5, 8, 0.9, 4, 5],
    [7, 6, 0.95, 5, 4],
    [9, 5, 0.9, 6, 3]
])

# y, 실제 성적
y = np.array([40, 55, 70, 85, 95])

m, n = X.shape
print(f"Dataset size: {m}, features: {n}")

# 데이터 시각화
# 특징 이름 지정
feature_names = ['Study Hours', 'Sleep Hours', 'Attendance ratio', 'Practice exam', 'Stress Level']

# 각 특징별로 성적(y)과의 관계 시각화
plt.figure(figsize=(12, 4))
for i in range(n):
    plt.subplot(1, n, i + 1)
    plt.scatter(X[:, i], y, color='red')
    plt.xlabel(feature_names[i])
    plt.ylabel('Score')
    plt.title(f'{feature_names[i]} vs Score')
plt.tight_layout()
plt.show()

# 비용 함수
def compute_cost(X, y, W, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb = np.dot(X[i], W) + b
        cost += (f_wb - y[i]) ** 2
    return cost / (2 * m)

# 기울기 계산
def compute_gradient(X, y, W, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        f_wb = np.dot(X[i], W) + b
        err = f_wb - y[i]
        dj_dw += err * X[i]
        dj_db += err

    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# 경사하강법
def gradient_descent(X, y, W, b, alpha, num_iters):    
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, W, b)
        W -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 100 == 0:
            cost = compute_cost(X, y, W, b)
            J_history.append(cost)
            print(f"iter {i}: cost={cost:.4f}, w={W}, b={b:.4f}")
    return W, b

# 초기값
W = np.zeros(X.shape[1])
b = 0
alpha = 0.01
num_iters = 1000

# 학습
w_final, b_final = gradient_descent(X, y, W, b, alpha, num_iters)
print(f'Final w: {w_final}, Final b: {b_final}')

# 예측
def predict(W, b, X):
    return np.dot(X, W) + b

X_test = np.array([
    # [1, 6, 0.7, 1, 8],
    # [3, 7, 0.8, 2, 6],
    # [5, 8, 0.9, 4, 5],
    # [7, 6, 0.95, 5, 4],
    # [9, 5, 0.9, 6, 3],
    [6, 7, 0.9, 4, 5],  # 테스트 입력 [공부시간, 수면시간, 출석률, 모의고사횟수, 스트레스]
])

for x in X_test:
    y_pred = predict(w_final, b_final, x)
    print(f'X= {x[0]}, {x[1]}, {x[2]}, {x[3]}, {x[4]} y= {y_pred:.2f}')
