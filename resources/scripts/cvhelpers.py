"""Shared helpers for the Machine Vision course notebooks.

Usage (after the setup cell has cloned the repository):

    import sys
    sys.path.append("resources/scripts")
    from cvhelpers import show, concept_map, image_grid, interactive
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

try:
    import ipywidgets as widgets
    from ipywidgets import interact
    HAS_WIDGETS = True
except Exception:
    HAS_WIDGETS = False


def show(*images, titles=None, cmap=None, figsize=None):
    """Display one or more images (BGR or gray) side by side."""
    titles = titles or [""] * len(images)
    figsize = figsize or (5 * len(images), 5)
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        if img is None:
            plt.text(0.5, 0.5, "None", ha="center", va="center")
        elif img.ndim == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap=cmap or "gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def concept_map(steps, title="Concept map", color="#e8f0fe", edge="#1a73e8"):
    """Draw a vertical flow diagram (infographic) from a list of steps."""
    n = len(steps)
    fig, ax = plt.subplots(figsize=(9, 0.95 * n + 1))
    ax.axis("off")
    ys = np.linspace(0.92, 0.08, n)
    for i, s in enumerate(steps):
        ax.text(0.5, ys[i], s, ha="center", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.55", fc=color, ec=edge, lw=1.6),
                wrap=True)
        if i < n - 1:
            ax.annotate("", xy=(0.5, ys[i + 1] + 0.055), xytext=(0.5, ys[i] - 0.055),
                        arrowprops=dict(arrowstyle="-|>", color="#5f6368", lw=1.8))
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def image_grid(images, cols=3, titles=None, figsize=(15, 5)):
    """Display a grid of images (useful for many small results)."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    titles = titles or [""] * n
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if img.ndim == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap="gray")
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_histogram(image, title="Histogram"):
    """Plot the intensity histogram of a gray or colour image."""
    plt.figure(figsize=(6, 4))
    if image.ndim == 2:
        plt.plot(cv2.calcHist([image], [0], None, [256], [0, 256]), color="k")
    else:
        for i, c in enumerate(["b", "g", "r"]):
            plt.plot(cv2.calcHist([image], [i], None, [256], [0, 256]), color=c)
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Pixel count")
    plt.xlim([0, 256])
    plt.show()
