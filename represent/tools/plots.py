import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.metrics import confusion_matrix, classification_report


def plot_img_grid(batch, row=4, col=4, ch_selector=[3, 2, 1], cmap=None):
    fig = plt.figure(figsize=(2 * row, 2 * col))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(row, col),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )

    for i in range(row * col):
        img = batch[i, ch_selector, :, :].numpy().transpose(1, 2, 0)
        grid[i].imshow(img, cmap=cmap)

    return grid


def print_classification_report(y_true, y_pred, classnames=None):
    report = classification_report(y_true, y_pred, target_names=classnames, zero_division=0)
    print(report)
    return report


def plot_confusion_matrix(
    y_true,
    y_pred,
    classnames=None,
    figsize=(10, 10),
    colorbar: bool = True,
):
    present_classes = np.unique(y_true)
    present_classnames = [classnames[c] for c in present_classes]

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize="true", labels=present_classes)
    cm_all = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None, labels=present_classes)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(np.arange(len(present_classes)))
    ax.set_yticks(np.arange(len(present_classes)))
    ax.set_xticklabels(present_classnames, rotation=30, ha="right")
    ax.set_yticklabels(present_classnames)

    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            v = f"{cm_all[c,r]}"
            v = f"{v[:-3]}k" if len(v) > 3 else v
            color = "black" if cm[c, r] < 0.5 else "white"
            ax.text(r, c, v, ha="center", va="center", color=color)

    ax.grid(False)
    ax.set_xlabel("predicted", labelpad=-600)
    ax.set_ylabel("true", loc="center", labelpad=-690)

    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, location="right", anchor=(0, 0.3), shrink=0.9)
        cbar.ax.set_title("recall")

    return ax
