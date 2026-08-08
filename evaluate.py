import argparse
import json

from src.env_setup import ensure_openmp_env

ensure_openmp_env()

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CATEGORIES, OUTPUTS_ROOT
from src.dataset import MVTecDataset
from src.metrics import best_f1_threshold, evaluate_category
from src.patchcore import PatchCore
from src.visualize import plot_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PatchCore model on the MVTec-AD test split.")
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-visualizations", type=int, default=8)
    args = parser.parse_args()

    model = PatchCore.load(args.category, device=args.device)

    test_dataset = MVTecDataset(args.category, split="test", image_size=model.config.image_size, crop_size=model.config.crop_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"[{args.category}] {len(test_dataset)} Testbilder geladen.")

    all_labels, all_image_scores, all_masks, all_score_maps = [], [], [], []
    all_images, all_defect_types = [], []

    for batch in tqdm(test_loader, desc="Evaluating"):
        image_scores, score_maps = model.predict(batch["image"])
        all_labels.append(batch["label"].numpy())
        all_image_scores.append(image_scores)
        all_masks.append(batch["mask"].numpy())
        all_score_maps.append(score_maps[:, None, :, :])
        all_images.append(batch["image"].numpy())
        all_defect_types.extend(batch["defect_type"])

    labels = np.concatenate(all_labels)
    image_scores = np.concatenate(all_image_scores)
    masks = np.concatenate(all_masks)
    score_maps = np.concatenate(all_score_maps)
    images = np.concatenate(all_images)

    metrics = evaluate_category(labels, image_scores, masks, score_maps)
    print(f"\n[{args.category}] Ergebnisse:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    threshold = best_f1_threshold(labels, image_scores)
    metrics["threshold"] = threshold
    print(f"  threshold (best F1): {threshold:.4f}")

    out_dir = OUTPUTS_ROOT / args.category
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(model.config.model_dir() / "threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    defect_indices = [i for i, l in enumerate(labels) if l == 1][: args.num_visualizations]
    good_indices = [i for i, l in enumerate(labels) if l == 0][: max(2, args.num_visualizations // 4)]
    for i in defect_indices + good_indices:
        plot_result(
            images[i],
            masks[i],
            score_maps[i, 0],
            image_scores[i],
            title=f"{args.category} / {all_defect_types[i]}",
            save_path=out_dir / f"vis_{i:03d}_{all_defect_types[i]}.png",
        )
    print(f"[{args.category}] Visualisierungen und Metriken gespeichert unter {out_dir}/")


if __name__ == "__main__":
    main()
