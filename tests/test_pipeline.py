"""End-to-end smoke test that runs the full PatchCore pipeline on synthetic
images, without needing the real (license-gated) MVTec-AD dataset. Run with:

    ./venv/bin/python tests/test_pipeline.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import ensure_openmp_env

ensure_openmp_env()

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from src.config import PatchCoreConfig
from src.dataset import MVTecDataset
from src.metrics import evaluate_category
from src.patchcore import PatchCore

CATEGORY = "synthetic_test"
IMAGE_SIZE = 96
CROP_SIZE = 96


def make_image(defect: bool, size: int = 128, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = (rng.normal(loc=140, scale=10, size=(size, size, 3))).clip(0, 255).astype("uint8")
    mask = np.zeros((size, size), dtype="uint8")
    if defect:
        arr[40:60, 40:60] = [255, 0, 0]
        mask[40:60, 40:60] = 255
    return Image.fromarray(arr), Image.fromarray(mask)


def build_synthetic_dataset(root: Path):
    train_dir = root / CATEGORY / "train" / "good"
    test_good_dir = root / CATEGORY / "test" / "good"
    test_defect_dir = root / CATEGORY / "test" / "scratch"
    gt_dir = root / CATEGORY / "ground_truth" / "scratch"
    for d in (train_dir, test_good_dir, test_defect_dir, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i in range(12):
        img, _ = make_image(defect=False, seed=i)
        img.save(train_dir / f"{i:03d}.png")

    for i in range(4):
        img, _ = make_image(defect=False, seed=100 + i)
        img.save(test_good_dir / f"{i:03d}.png")

    for i in range(4):
        img, mask = make_image(defect=True, seed=200 + i)
        img.save(test_defect_dir / f"{i:03d}.png")
        mask.save(gt_dir / f"{i:03d}_mask.png")


def main():
    tmp_root = Path("/tmp/patchcore_smoke_test_data")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    build_synthetic_dataset(tmp_root)

    train_dataset = MVTecDataset(CATEGORY, split="train", data_root=tmp_root, image_size=IMAGE_SIZE, crop_size=CROP_SIZE)
    test_dataset = MVTecDataset(CATEGORY, split="test", data_root=tmp_root, image_size=IMAGE_SIZE, crop_size=CROP_SIZE)
    assert len(train_dataset) == 12
    assert len(test_dataset) == 8
    print(f"Dataset OK: {len(train_dataset)} train / {len(test_dataset)} test Bilder.")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    config = PatchCoreConfig(category=CATEGORY, coreset_ratio=0.5, image_size=IMAGE_SIZE, crop_size=CROP_SIZE)
    model = PatchCore(config)
    model.fit(train_loader)
    assert model.memory_bank is not None and model.memory_bank.shape[0] > 0
    print(f"Fit OK: Memory Bank mit {model.memory_bank.shape[0]} Vektoren.")

    all_labels, all_scores, all_masks, all_score_maps = [], [], [], []
    for batch in test_loader:
        image_scores, score_maps = model.predict(batch["image"])
        all_labels.append(batch["label"].numpy())
        all_scores.append(image_scores)
        all_masks.append(batch["mask"].numpy())
        all_score_maps.append(score_maps[:, None, :, :])

    labels = np.concatenate(all_labels)
    scores = np.concatenate(all_scores)
    masks = np.concatenate(all_masks)
    score_maps = np.concatenate(all_score_maps)
    assert scores.shape[0] == 8
    assert score_maps.shape[-2:] == (CROP_SIZE, CROP_SIZE)
    print(f"Predict OK: scores shape={scores.shape}, score_maps shape={score_maps.shape}")

    metrics = evaluate_category(labels, scores, masks, score_maps)
    print(f"Metrics OK: {metrics}")
    assert 0.0 <= metrics["image_auroc"] <= 1.0
    assert 0.0 <= metrics["pixel_auroc"] <= 1.0
    assert 0.0 <= metrics["pro_score"] <= 1.0

    model.save()
    reloaded = PatchCore.load(CATEGORY)
    image_scores2, _ = reloaded.predict(next(iter(test_loader))["image"])
    assert image_scores2.shape[0] == 4
    print("Save/Load OK.")

    shutil.rmtree(tmp_root)
    shutil.rmtree(config.model_dir())
    print("\nALLE SMOKE TESTS BESTANDEN.")


if __name__ == "__main__":
    main()
