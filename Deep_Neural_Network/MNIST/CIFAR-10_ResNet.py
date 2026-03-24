from torchvision.datasets import CIFAR10
from torchvision import transforms

tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

tf_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
])

cf_train = CIFAR10(
    root='./data',
    train=True,
    transform=tf_train,
    download=True
)

cf_test = CIFAR10(
    root='./data',
    train=False,
    transform=tf_test,
    download=True
)

print(cf_train.data.shape)

# from google.colab import files
# files.upload()

# from MNIST_func import plot_CIFAR10
# plot_CIFAR10(cf_train, 0, 20)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bt_size = 64
    pin_mode = False
    workers = 0
    print(f'device={device}')

    if torch.cuda.is_available():
        bt_size *= 2
        pin_mode = True
        workers = 2
        print(f'gpu={torch.cuda.get_device_name()}')

    train_dl = DataLoader(cf_train, batch_size=bt_size, shuffle=True, pin_memory=pin_mode, num_workers=workers)

    class ResidualBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride=1):
            super().__init__()
            # 논문 방식: 3x3 Conv -> BN -> ReLU -> 3x3 Conv -> BN
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
            
            # 논문의 identity shortcut, 항등행렬()
            # out = F(x) + x
            self.shortcut = nn.Sequential()

            # 입력과 출력의 크기/채널이 다를 때 맞춰주는 1x1 Conv             
            # out = self.conv(x)는 F(x)
            # out += shortcut(x)는 skip, F(x) + x 를 위해 텐서간 (C, H, W) 가 같아야 함            
            # Case1 : stride=2, Fx=(64, 16, 16), x=(64, 32, 32) 더할 수 없음
            # Case2 : in_ch!=out_ch & stride=1, F(x)=(128, 32, 32), x=(64, 32, 32),  더할 수 없음

            if stride != 1 or in_ch != out_ch:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch)
                )

            self.relu = nn.ReLU()

        def forward(self, x):
            out = self.conv(x)
            out += self.shortcut(x) # Skip Connection
            return self.relu(out)

    # 1. 입력   (3, 32, 32)
    # 2. conv1  (16, 32, 32)
    # 3. stage1 (16, 32, 32)
    # 4. state2 (32, 16, 16)
    # 5. state3 (64, 8, 8)
    # 6. avg    (64, 1, 1)
    # 7. fc     (64*1*1, )
    # 8. 출력   (64, 10)

    class ResNet_CIFAR(nn.Module):
        def __init__(self, num_blocks=[3, 3, 3]): # 각 스테이지별 블록 개수
            super().__init__()
            self.in_ch = 16
            
            # 1. Stem: 초반 특징 추출 (논문 CIFAR-10용은 16채널로 시작)
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
            
            # 2. Stages: 채널을 16 -> 32 -> 64로 늘리며 해상도를 줄임
            self.stage1 = self._make_layer(16, num_blocks[0], stride=1)
            self.stage2 = self._make_layer(32, num_blocks[1], stride=2)
            self.stage3 = self._make_layer(64, num_blocks[2], stride=2)
            
            # 3. Output: Global Average Pooling 후 FC
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 10)

        def _make_layer(self, out_ch, num_blocks, stride):
             # stage1 (out_ch=16, num_blocks=3, stride=1) = [1] + [1] * (3 - 1) = [1, 1, 1]
             # stage2 (out_ch=32, num_blocks=3, stride=2) = [2] + [1] * (3 - 1) = [2, 1, 1]
             # stage3 (out_ch=64, num_blocks=3, stride=2) = [2] + [1] * (3 - 1) = [2, 1, 1]
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for s in strides:
                layers.append(ResidualBlock(self.in_ch, out_ch, s))
                self.in_ch = out_ch
            return nn.Sequential(*layers) # *:unpacking Sequential(layer1, layer2, layer2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        
    model = ResNet_CIFAR().to(device)
    criterion = nn.CrossEntropyLoss()

    # monentum : 이전의 기울기를 90% 기억
    # v     = 0.9 * (이전 v) + (현재 gradient)
    # w = w - lr * v

    # weight_decay : L2 정규화, 람다와 같다 0.005
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    epochs = 200

    for i in range(epochs):
        model.train()
        cost = 0

        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cost += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        cost /= len(train_dl)
        print(f'epoch={i}, cost={cost:.3f}, lr={current_lr:.3f}')    

    # predict
    model.eval()
    with torch.no_grad():
        test_dl = DataLoader(cf_test, batch_size=128)
        cnt=0

        for x_batch, y_batch in test_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            Z = torch.softmax(y_pred, dim=1)
            A = torch.argmax(Z, dim=1)

            cnt += (A==y_batch).sum().item()

        acc = cnt / len(cf_test)
        print(f'Accuracy={acc:.3f}')

    # result : 0.921

