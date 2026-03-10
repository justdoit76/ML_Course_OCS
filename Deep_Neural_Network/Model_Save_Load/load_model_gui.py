import sys
import numpy as np
import torch
import torch.nn as nn
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtGui import QPainter, QPen, QImage, QColor
from PyQt6.QtCore import Qt, QPoint
from PIL import Image

# 1. 학습한 모델 구조 (기존 학습모델 구조와 동일하게 수정)
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

# 2. 그림판 위젯
class Canvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(280, 280)
        self.image = QImage(self.size(), QImage.Format.Format_Grayscale8)
        self.image.fill(Qt.GlobalColor.black)
        self.pt = QPoint()

    def paintEvent(self, e):
        qp = QPainter(self)
        qp.drawImage(0, 0, self.image)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.pt = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            qp = QPainter(self.image)
            p = QPen(Qt.GlobalColor.white, 20, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            qp.setPen(p)
            pt = e.position().toPoint()
            qp.drawLine(self.pt, pt)
            self.pt = pt
            self.update()

    def onClear(self):
        self.image.fill(Qt.GlobalColor.black)
        self.update()

    def getImage(self):
        # QImage를 PIL Image로 변환하여 28x28 리사이징
        ptr = self.image.bits()
        ptr.setsize(self.image.sizeInBytes())
        arr = np.frombuffer(ptr, np.uint8).reshape((280, 280))
        img = Image.fromarray(arr).resize((280, 280)).resize((28, 28), Image.Resampling.LANCZOS)
        return img

# 3. 메인 윈도우
class MNISTApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MNIST Predictor')
        
        # 모델 로드 (파일 경로를 본인의 .pth 파일로 수정)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load('mnist_model.pth', map_location=self.device))
            self.model.eval()
            print('모델 로드 성공!')
        except:
            print('모델 찾기 실패!')

        self.initUi()

    def initUi(self):
        vbox = QVBoxLayout()
        
        self.canvas = Canvas()
        self.label = QLabel('결과: 대기 중')
        self.label.setStyleSheet('font-size: 20px; font-weight: bold;')
        
        hbox = QHBoxLayout()
        self.btn1 = QPushButton('예측하기')
        self.btn2 = QPushButton('지우기')
        
        self.btn1.clicked.connect(self.onPredict)
        self.btn2.clicked.connect(self.canvas.onClear)
        
        hbox.addWidget(self.btn1)
        hbox.addWidget(self.btn2)
        
        vbox.addWidget(QLabel('마우스로 숫자를 그리세요 (280x280 -> 28x28)'))
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.label)
        vbox.addLayout(hbox)
        
        w = QWidget()
        w.setLayout(vbox)
        self.setCentralWidget(w)

    def onPredict(self):
        img = self.canvas.getImage()
        # 텐서 변환 및 정규화 (0~1 사이 값)
        np_arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.FloatTensor(np_arr).unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            y_pred = self.model(tensor)
            _class = torch.argmax(y_pred, dim=1).item()
            prob_all = torch.softmax(y_pred, dim=1)            
            prob_target = prob_all[0][_class] * 100
            print(prob_all[0])
            
        self.label.setText(f'결과: {_class} ({prob_target:.2f}%)')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MNISTApp()
    window.show()
    sys.exit(app.exec())