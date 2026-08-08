import argparse
import time

from src.env_setup import ensure_openmp_env

ensure_openmp_env()

from torch.utils.data import DataLoader

from src.config import CATEGORIES, PatchCoreConfig
from src.dataset import MVTecDataset
from src.patchcore import PatchCore


def main():
    parser = argparse.ArgumentParser(description="Train a PatchCore memory bank for one category.")
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--coreset-ratio", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = PatchCoreConfig(category=args.category, coreset_ratio=args.coreset_ratio, device=args.device)

    train_dataset = MVTecDataset(args.category, split="train", image_size=config.image_size, crop_size=config.crop_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"[{args.category}] {len(train_dataset)} Trainingsbilder geladen.")

    model = PatchCore(config)

    start = time.time()
    model.fit(train_loader)
    elapsed = time.time() - start

    model.save()
    print(
        f"[{args.category}] Memory Bank gebaut in {elapsed:.1f}s, "
        f"{model.memory_bank.shape[0]} Coreset-Vektoren (aus {int(model.memory_bank.shape[0] / config.coreset_ratio)} Patches), "
        f"gespeichert unter models/{args.category}/"
    )


if __name__ == "__main__":
    main()
