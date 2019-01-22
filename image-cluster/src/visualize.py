"""
t-SNE 를 사용한 visualization 모듈입니다.
"""
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from config import *


def visualize():
    # loading datasets
    features = np.load(os.path.join(DATA_DIR, FEATURES + ".npy"))
    labels_true = np.load(os.path.join(DATA_DIR, LABELS_TURE + ".npy"))
    labels_pred = np.load(os.path.join(DATA_DIR, LABELS_PRED + ".npy"))

    # plot true labels
    tsne = TSNE(learning_rate=100, verbose=1)
    # fitting model
    transformed = tsne.fit_transform(features)

    # plotting 2d t-Sne
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]

    # show true labels
    labels_true_uniq = list(set(labels_true))
    z = range(1, len(labels_true_uniq))
    hsv = plt.get_cmap('hsv')
    colors_norm = colors.Normalize(vmin=0, vmax=len(labels_true_uniq))
    scalar_map = cmx.ScalarMappable(norm=colors_norm, cmap=hsv)

    fig = plt.figure()
    ax_true = fig.add_subplot(211)
    ax_pred = fig.add_subplot(212)

    for i in range(len(labels_true_uniq)):
        idx = labels_true == labels_true_uniq[i]
        ax_true.scatter(x_axis[idx], y_axis[idx], c=scalar_map.to_rgba(i), s=10)
    ax_true.set_title("t-SNE visualization of clustering colored by true labels")

    # show predicted labels
    labels_pred_uniq = list(set(labels_pred))
    z = range(1, len(labels_pred_uniq))
    hsv = plt.get_cmap('hsv')
    colors_norm = colors.Normalize(vmin=0, vmax=len(labels_pred_uniq))
    scalar_map = cmx.ScalarMappable(norm=colors_norm, cmap=hsv)

    for i in range(len(labels_pred_uniq)):
        idx = labels_pred == labels_pred_uniq[i]
        ax_pred.scatter(x_axis[idx], y_axis[idx], c=scalar_map.to_rgba(i), s=10)
    ax_pred.set_title("t-SNE visualization of clustering colored by predicted labels")

    plt.show()


if __name__ == '__main__':
    visualize()
