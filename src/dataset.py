from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import DATA_ROOT, IMAGENET_MEAN, IMAGENET_STD

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def build_transform(image_size: int, crop_size: int, mask: bool = False):
    if mask:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class MVTecDataset(Dataset):
    """Expects the standard MVTec-AD folder layout:

    data/<category>/train/good/*.png
    data/<category>/test/<good|defect_type>/*.png
    data/<category>/ground_truth/<defect_type>/*_mask.png
    """

    def __init__(self, category: str, split: str, data_root: Path = DATA_ROOT,
                 image_size: int = 256, crop_size: int = 224):
        assert split in ("train", "test")
        self.category = category
        self.split = split
        self.root = Path(data_root) / category
        if not self.root.exists():
            raise FileNotFoundError(
                f"Kein Datensatz gefunden unter {self.root}. "
                f"Siehe README.md fuer die Download-Anleitung."
            )

        self.image_transform = build_transform(image_size, crop_size, mask=False)
        self.mask_transform = build_transform(image_size, crop_size, mask=True)

        self.samples = []
        if split == "train":
            good_dir = self.root / "train" / "good"
            for p in sorted(good_dir.iterdir()):
                if p.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append({"image": p, "label": 0, "mask": None, "defect_type": "good"})
        else:
            test_dir = self.root / "test"
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_type = defect_dir.name
                is_good = defect_type == "good"
                for p in sorted(defect_dir.iterdir()):
                    if p.suffix.lower() not in IMG_EXTENSIONS:
                        continue
                    mask_path = None
                    if not is_good:
                        mask_path = self.root / "ground_truth" / defect_type / f"{p.stem}_mask.png"
                        if not mask_path.exists():
                            mask_path = None
                    self.samples.append(
                        {"image": p, "label": 0 if is_good else 1, "mask": mask_path, "defect_type": defect_type}
                    )

        if not self.samples:
            raise RuntimeError(f"Keine Bilder in {self.root} ({split}) gefunden.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        image_t = self.image_transform(image)

        if sample["mask"] is not None:
            mask = Image.open(sample["mask"]).convert("L")
            mask_t = self.mask_transform(mask)
            mask_t = (mask_t > 0.5).float()
        else:
            crop = self.image_transform.transforms[1].size
            size = crop if isinstance(crop, int) else crop[0]
            mask_t = torch.zeros((1, size, size))

        return {
            "image": image_t,
            "label": sample["label"],
            "mask": mask_t,
            "defect_type": sample["defect_type"],
            "path": str(sample["image"]),
        }


def available_categories(data_root: Path = DATA_ROOT):
    if not data_root.exists():
        return []
    return sorted(p.name for p in data_root.iterdir() if p.is_dir() and (p / "train" / "good").exists())
