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


print(X.shape)
# y, 실제 성적
y = np.array([40, 55, 70, 85, 95])

# Z score
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
Xp = (X-mu)/sigma

m, n = Xp.shape
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

from Linear_Func import gradient_descent

# 초기값
W = np.zeros(n)
print(W)
b = 0.0
lr = 0.001
epochs = 10000
Lambda = 0

# 학습
W_final, b_final = gradient_descent(Xp, y, W, b, lr, epochs, Lambda)
print(f'Final w: {np.round(W_final, 3)}, Final b: {b_final:.3f}')

# 예측
def predict(X, W, b):
    return np.dot(X, W) + b

X_test = np.array([
    # [1, 6, 0.7, 1, 8],
    # [3, 7, 0.8, 2, 6],
    # [5, 8, 0.9, 4, 5],
    # [7, 6, 0.95, 5, 4],
    # [9, 5, 0.9, 6, 3],
    [6, 7, 0.9, 4, 5],  # 테스트 입력 [공부시간, 수면시간, 출석률, 모의고사횟수, 스트레스]
])

X_test_p = (X_test-mu)/sigma

y_pred = predict(X_test_p, W_final, b_final)
for i in range(len(X_test)):
    print(f'X_test={X_test[i]} y_pred={y_pred[i]:.3f}')
