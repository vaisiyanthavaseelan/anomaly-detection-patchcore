"""Builds the README hero image: a compact before/after grid for one
metal_nut and one screw defect example, meant to communicate the result to a
non-technical reader (recruiter) at a glance, without reading any code.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import ensure_openmp_env

ensure_openmp_env()

import json

import matplotlib.pyplot as plt

from src.dataset import MVTecDataset
from src.patchcore import PatchCore
from src.visualize import denormalize

EXAMPLES = [
    {"category": "metal_nut", "defect_type": "bent", "index_in_type": 0, "label_de": "Metallmutter: verbogen"},
    {"category": "screw", "defect_type": "scratch_neck", "index_in_type": 0, "label_de": "Schraube: Kratzer"},
]


def find_sample(dataset, defect_type, index_in_type):
    matches = [i for i, s in enumerate(dataset.samples) if s["defect_type"] == defect_type]
    return matches[index_in_type]


def main():
    fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(8, 4 * len(EXAMPLES)))

    for row, example in enumerate(EXAMPLES):
        category = example["category"]
        model = PatchCore.load(category)
        dataset = MVTecDataset(category, split="test", image_size=model.config.image_size, crop_size=model.config.crop_size)
        idx = find_sample(dataset, example["defect_type"], example["index_in_type"])
        sample = dataset[idx]

        image_scores, score_maps = model.predict(sample["image"].unsqueeze(0))
        threshold = json.load(open(model.config.model_dir() / "threshold.json"))["threshold"]
        score = image_scores[0]
        is_anomaly = score > threshold

        img = denormalize(sample["image"].numpy())

        ax_orig, ax_overlay = axes[row]
        ax_orig.imshow(img)
        ax_orig.set_title(example["label_de"], fontsize=13, fontweight="bold")
        ax_orig.axis("off")

        ax_overlay.imshow(img)
        ax_overlay.imshow(score_maps[0], cmap="jet", alpha=0.5)
        verdict = "DEFEKT ERKANNT" if is_anomaly else "OK"
        color = "#c0392b" if is_anomaly else "#27ae60"
        ax_overlay.set_title(f"{verdict}  (Score {score:.1f} / Schwelle {threshold:.1f})", fontsize=12, color=color, fontweight="bold")
        ax_overlay.axis("off")

    fig.suptitle("Automatische Fehlererkennung ohne Beispiel-Defekte im Training", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = Path(__file__).resolve().parent.parent / "assets" / "hero.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Hero-Bild gespeichert: {out_path}")


if __name__ == "__main__":
    main()
