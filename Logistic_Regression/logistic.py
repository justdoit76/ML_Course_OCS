import numpy as np

X = np.array([1, 3, 5, 7, 9])
y = np.array([0,0,0,1,1])

W = 0
b = 0
lr = 0.01
epochs = 3000
_lambda = 0

from logistic_func import gradient_descent, sigmoid
W_final, b_final = gradient_descent(X, y, W, b, lr, epochs, _lambda)

def predict(X, W, b):
    z = W*X+b
    return sigmoid(z)

# predict
X_test = 6
y_pred = predict(X_test, W_final, b_final)
result = 'Pass' if y_pred >=0.5 else 'Fail'
print(f'X_test={X_test}, y_pred={y_pred:.2f}, Result={result}')

# 그래프 설정
import matplotlib.pyplot as plt
X_range = np.linspace(0, 10, 100)
y_range = predict(X_range, W_final, b_final)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', s=100, label='Input', zorder=5)

# 모델이 예측한 시그모이드 곡선 (파란 선)
plt.plot(X_range, y_range, color='blue', linewidth=2, label='Logistic Curve')

# 결정 경계선(확률 0.5 지점)
plt.axhline(y=0.5, color='gray', linestyle='--', label='Threshold (0.5)')
plt.axvline(x=-b/W if W != 0 else 0, color='green', linestyle=':', label='Decision Boundary')

# 테스트 포인트 (5시간 지점 표시)
y_test = predict(X_test, W_final, b_final)
plt.scatter(X_test, y_test, color='orange', marker='x', s=150, linewidth=3, label=f'X={X_test} (Prob={y_test:.2f})', zorder=6)

# 그래프 범례
plt.title('Logistic Regression: Study Hour vs Pass,Fail', fontsize=15)
plt.xlabel('Study Hours (X)', fontsize=12)
plt.ylabel('Pro. Passing (y_hat)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()