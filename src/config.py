from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_ROOT = PROJECT_ROOT / "models"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

CATEGORIES = ["metal_nut", "screw"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class PatchCoreConfig:
    category: str
    backbone: str = "wide_resnet50_2"
    layers: tuple = ("layer2", "layer3")
    image_size: int = 256
    crop_size: int = 224
    pool_kernel: int = 3
    coreset_ratio: float = 0.01
    projection_dim: int = 128
    num_neighbors: int = 1
    reweight_k: int = 9
    device: str = "cpu"

    def model_dir(self) -> Path:
        d = MODELS_ROOT / self.category
        d.mkdir(parents=True, exist_ok=True)
        return d
