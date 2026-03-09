import torch
import matplotlib.pyplot as plt
import math

def plot_CIFAR10(dataset, start, end, cols=5):
    n = end - start
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()

    for i in range(start, end):
        img, label = dataset[i]
        j = i - start   

        # Tensor → numpy
        if isinstance(img, torch.Tensor):
            img = img.numpy()

        # (C,H,W) → (H,W,C)
        img = img.transpose(1, 2, 0)

        axes[j].imshow(img)
        axes[j].set_title(str(label))
        axes[j].axis('off')

    # 남는 subplot 숨기기
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_CIFAR10_Neurons(model, img, device):
    # 배치 차원 추가 → (1,3,32,32)
    x = img.unsqueeze(0).to(device)

    with torch.no_grad():
        # conv block
        conv1 = model.conv[:4](x)     # Conv+BN+ReLU+Pool → (1,16,16,16)
        conv2 = model.conv[4:8](conv1) # → (1,32,8,8)
        conv3 = model.conv[8:](conv2)  # → (1,64,8,8)

    conv1 = conv1.cpu()
    conv2 = conv2.cpu()
    conv3 = conv3.cpu()
    x = x.cpu()

    # 입력 이미지 (RGB)
    plt.figure(figsize=(3,3))
    plt.title("Input Image")
    plt.imshow(x[0].permute(1,2,0).numpy())  # (C,H,W) → (H,W,C)
    plt.axis('off')
    plt.show()

    # Conv1 feature maps (16개)
    fig, axes = plt.subplots(4,4, figsize=(8,8))
    fig.suptitle("Conv1 Feature Maps")

    for i, ax in enumerate(axes.flat):
        ax.imshow(conv1[0,i].numpy(), cmap='gray')
        ax.axis('off')

    plt.show()

    # Conv2 feature maps (32개)
    fig, axes = plt.subplots(4,8, figsize=(12,6))
    fig.suptitle("Conv2 Feature Maps")

    for i, ax in enumerate(axes.flat):
        ax.imshow(conv2[0,i].numpy(), cmap='gray')
        ax.axis('off')

    plt.show()

    # Conv3 feature maps (64개 → 일부만)
    fig, axes = plt.subplots(8,8, figsize=(12,12))
    fig.suptitle("Conv3 Feature Maps (64)")

    for i, ax in enumerate(axes.flat):
        ax.imshow(conv3[0,i].numpy(), cmap='gray')
        ax.axis('off')

    plt.show()