from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import IMAGENET_MEAN, IMAGENET_STD


def denormalize(image_t: np.ndarray) -> np.ndarray:
    """(C, H, W) normalized tensor -> (H, W, C) uint8-range float image."""
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = image_t * std + mean
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def plot_result(image_t: np.ndarray, gt_mask: np.ndarray, score_map: np.ndarray,
                 image_score: float, title: str, save_path: Path):
    img = denormalize(image_t)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original")

    axes[1].imshow(gt_mask.squeeze(), cmap="gray")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(score_map, cmap="jet")
    axes[2].set_title("Anomaly Heatmap")

    axes[3].imshow(img)
    axes[3].imshow(score_map, cmap="jet", alpha=0.5)
    axes[3].set_title(f"Overlay (score={image_score:.2f})")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
