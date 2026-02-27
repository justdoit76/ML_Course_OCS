import torch
import matplotlib.pyplot as plt
import math

def plot_MNIST(dataset, start, end, cols=5):
    n = end-start
    rows = math.ceil(n/cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = axes.flatten()

    for i in range(start, end):
        img, label = dataset[i]       

        j = i-start

        axes[j].imshow(img, cmap='gray')
        axes[j].set_title(f"{label}")
        axes[j].axis('off')

    # 남는 subplot 숨기기
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

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