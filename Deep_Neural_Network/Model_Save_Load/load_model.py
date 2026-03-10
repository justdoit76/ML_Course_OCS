import torch
import torch.nn as nn

# 이전에 학습한 모델 불러오기

# 1. 모델 구조 정의 (학습 때와 동일해야 함)
class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2), # (16, 14, 14)

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2), # (32, 7, 7)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, X):
        X = self.conv(X)
        X = self.fc(X)
        return X

# 2. 모델 생성
model = CNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. 가중치 불러오기
# 저장데이터는 W, b, BatchNorm 등
stateDict = torch.load('mnist_model.pth', map_location=device)
model.load_state_dict( stateDict )

# 4. 평가모드로 전환
model.eval()

# 5. 불러온 모델로 예측
with torch.no_grad():
    img, label = X_test[0]
    img = img.unsqueeze(0)  # batch 차원 추가

    output = model(img)
    pred = output.argmax(dim=1)

    print("예측:", pred.item())
    print("정답:", label)