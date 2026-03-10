import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

def make_grid_tensor(X, resolution, padding=0.5):
    x_min, x_max = X[:, 0].min()-padding, X[:, 0].max()+padding
    y_min, y_max = X[:, 1].min()-padding, X[:, 1].max()+padding
    
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    X_test = np.c_[gx.ravel(), gy.ravel()]
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    return gx, gy, X_test_tensor


def plot_decision_boundary(model, X, y, C):
    model.eval()
    with torch.no_grad():        
        gx, gy, X_test_tensor = make_grid_tensor(X, 300, 0.5)

        Z = model(X_test_tensor)
        A = torch.argmax(Z, dim=1)

        A = A.reshape(gx.shape)

        cmap = plt.get_cmap('tab10')
        bound = np.arange(C + 1) - 0.5
        norm = BoundaryNorm(bound, cmap.N)

        plt.contourf(gx, gy, A, levels=bound, cmap=cmap, norm=norm, alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, norm=norm, alpha=0.6)
        plt.colorbar(label='Class', ticks=range(C))
        plt.show()   


def get_submodel(model, until):
    submodel = nn.Sequential(*list(model.children())[:until])
    return submodel


def plot_hidden_layer_with_neurons(model, X, y, until, max_neurons=None, cols=4, figsize=(10, 6)):
    model.eval()
    with torch.no_grad():
        gx, gy, X_test_tensor = make_grid_tensor(X, 300, 0.5)

        hLayer = get_submodel(model, until)
        H = hLayer(X_test_tensor)   # (N, hidden_dim)

        hidden_dim = H.shape[1]
        if max_neurons is not None:
            hidden_dim = min(hidden_dim, max_neurons)

        n_cols = min(cols, hidden_dim)
        n_rows = int(np.ceil(hidden_dim / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes)
        
        vmin = H[:, :hidden_dim].min().item()
        vmax = H[:, :hidden_dim].max().item()

        for i in range(hidden_dim):
            r, c = divmod(i, n_cols)
            ax = axes[r, c]

            activation = H[:, i].reshape(gx.shape)

            ax.contourf(
                gx, gy, activation,
                levels=30,
                cmap='plasma',
                vmin=vmin,
                vmax=vmax
            )

            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', s=8)
            ax.set_title(f'N{i}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # 빈 subplot 제거
        for j in range(hidden_dim, n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r, c].axis('off')
        
        
        fig.suptitle(f'Hidden layer activations (until={until})', fontsize=14)
        #plt.tight_layout()
        plt.show()