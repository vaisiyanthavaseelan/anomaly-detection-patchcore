import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class PatchFeatureExtractor(nn.Module):
    """Extracts locally-aware patch features from two mid-level backbone layers,
    following the PatchCore approach (Roth et al., 2022).

    layer2 keeps enough spatial resolution for pixel-level localization while
    layer3 adds more semantic context; layer1 is too low-level and layer4 too
    coarse/abstract for this.
    """

    def __init__(self, backbone: str = "wide_resnet50_2", layers=("layer2", "layer3"), pool_kernel: int = 3):
        super().__init__()
        if backbone != "wide_resnet50_2":
            raise ValueError(f"Unsupported backbone: {backbone}")

        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V2
        self.backbone = wide_resnet50_2(weights=weights)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        self.layers = layers
        self.pool_kernel = pool_kernel
        self._features = {}
        for name in layers:
            getattr(self.backbone, name).register_forward_hook(self._make_hook(name))

    def _make_hook(self, name):
        def hook(_module, _input, output):
            self._features[name] = output
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns a (B, C, H, W) patch-feature map at the resolution of the
        first configured layer (typically layer2)."""
        self._features = {}
        self.backbone(x)

        maps = [self._features[name] for name in self.layers]
        pooled = [
            F.avg_pool2d(m, kernel_size=self.pool_kernel, stride=1, padding=self.pool_kernel // 2)
            for m in maps
        ]

        target_size = pooled[0].shape[-2:]
        aligned = [pooled[0]] + [
            F.interpolate(m, size=target_size, mode="bilinear", align_corners=False) for m in pooled[1:]
        ]
        return torch.cat(aligned, dim=1)

    @staticmethod
    def flatten_patches(feature_map: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B*H*W, C)"""
        b, c, h, w = feature_map.shape
        return feature_map.permute(0, 2, 3, 1).reshape(b * h * w, c)
