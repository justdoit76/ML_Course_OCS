import numpy as np

y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

y_mean = np.mean(y_true)

ss_res = np.sum((y_true - y_pred) ** 2)   # 잔차 제곱합, 예측값과 실제값의 차이, 모델의 오차 크기
ss_tot = np.sum((y_true - y_mean) ** 2)   # 전체 변동합, 실제값과 평균값의 차이, 데이터 자체의 분산

r2 = 1 - (ss_res / ss_tot)
print(f'R2 = {r2:.3f}')

# 선형회귀 평가지표 R2_Score
# R2 값 : 모델이 데이터의 분산을 얼마나 설명했는지 비율로 나타낸 값
# 1.0   완벽하게 일치
# 0.8   데이터 변동의 80%를 모델이 설명함
# 0     모델이 평균값으로 예측한 것과 같음
# 음수  모델이 평균값보다도 못함