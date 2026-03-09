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

        # Tensor → numpy
        if isinstance(img, torch.Tensor):
            img = img.numpy()

        # (C, H, W) -> (H, W, C)
        img = img.transpose(1, 2, 0)     

        j = i-start

        axes[j].imshow(img, cmap='gray')
        axes[j].set_title(f"{label}")
        axes[j].axis('off')

    # 남는 subplot 숨기기
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def plot_MNIST_Neurons(model, img, device):    
    x = img.unsqueeze(0).to(device)
    print(x.shape)

    with torch.no_grad(): 
        # input x -> (batch, channel, H, W) = (1, 1, 28, 28)

        # conv1
        conv1 = model.conv[0](x)        # conv (1, 16, 28, 28)
        conv1 = model.conv[1](conv1)    # relu (1, 16, 28, 28)
        conv1 = model.conv[2](conv1)    # maxp (1, 16, 14, 14)

        # conv2
        conv2 = model.conv[3](conv1)    # conv (1, 32, 14, 14)   
        conv2 = model.conv[4](conv2)    # relu (1, 32, 14, 14)
        conv2 = model.conv[5](conv2)    # maxp (1, 32,  7,  7)

    # Tensor GPU -> CPU 이동, matplotlib는 GPU접근 (X)
    conv1 = conv1.cpu()
    conv2 = conv2.cpu()
    x = x.cpu()
    
    # 입력 이미지
    plt.figure(figsize=(3,3))
    plt.title("Input Image")
    plt.imshow(x[0,0].cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.show()
    
    # Conv1 feature map
    fig, axes = plt.subplots(4,4, figsize=(8,8))
    fig.suptitle("Conv1 Feature Maps")

    for i, ax in enumerate(axes.flat):
        ax.imshow(conv1[0,i].cpu().numpy(), cmap='gray')
        ax.axis('off')

    plt.show()

    # Conv2 feature map
    fig, axes = plt.subplots(4,8, figsize=(12,6))
    fig.suptitle("Conv2 Feature Maps")

    for i, ax in enumerate(axes.flat):
        ax.imshow(conv2[0,i].cpu().numpy(), cmap='gray')
        ax.axis('off')

    plt.show()