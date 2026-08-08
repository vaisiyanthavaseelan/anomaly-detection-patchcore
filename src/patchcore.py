import json
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from src.config import PatchCoreConfig
from src.feature_extractor import PatchFeatureExtractor
from src.memory_bank import GreedyCoresetSampler, extract_all_patch_features


class PatchCore:
    """End-to-end PatchCore model: feature extraction, memory bank with coreset
    subsampling, FAISS nearest-neighbor search, and image/pixel anomaly scoring.
    """

    def __init__(self, config: PatchCoreConfig):
        self.config = config
        self.extractor = PatchFeatureExtractor(
            backbone=config.backbone, layers=config.layers, pool_kernel=config.pool_kernel
        )
        self.index: faiss.Index | None = None
        self.memory_bank: torch.Tensor | None = None
        self.feature_map_size: tuple[int, int] | None = None

    def fit(self, train_dataloader):
        raw_features = extract_all_patch_features(self.extractor, train_dataloader, device=self.config.device)

        sampler = GreedyCoresetSampler(
            ratio=self.config.coreset_ratio,
            projection_dim=self.config.projection_dim,
            device=self.config.device,
        )
        self.memory_bank = sampler.sample(raw_features)
        self._build_index()

        sample_batch = next(iter(train_dataloader))
        with torch.no_grad():
            fmap = self.extractor(sample_batch["image"][:1].to(self.config.device))
        self.feature_map_size = tuple(fmap.shape[-2:])

    def _build_index(self):
        vectors = self.memory_bank.numpy().astype("float32")
        self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)

    @torch.no_grad()
    def predict(self, images: torch.Tensor):
        """images: (B, 3, H, W) -> (image_scores: (B,), score_maps: (B, H, W))"""
        images = images.to(self.config.device)
        feature_map = self.extractor(images)
        b, c, fh, fw = feature_map.shape
        patches = self.extractor.flatten_patches(feature_map).numpy().astype("float32")

        distances, neighbor_idx = self.index.search(patches, self.config.num_neighbors)
        patch_scores = np.sqrt(distances[:, 0])
        patch_scores = patch_scores.reshape(b, fh, fw)

        image_scores = np.array(
            [
                self._reweight_score(patch_scores[i], neighbor_idx.reshape(b, fh, fw, -1)[i], patches.reshape(b, fh, fw, -1)[i])
                for i in range(b)
            ]
        )

        score_maps_t = torch.from_numpy(patch_scores).unsqueeze(1)
        score_maps_up = F.interpolate(
            score_maps_t, size=images.shape[-2:], mode="bilinear", align_corners=False
        ).squeeze(1)

        return image_scores, score_maps_up.numpy()

    def _reweight_score(self, patch_score_map, neighbor_idx_map, patch_features_map):
        """PatchCore's confidence reweighting: down-weights the max patch score
        when its nearest memory-bank neighbor sits in a dense, well-covered
        region (i.e. is itself easy to match), making single outlier patches
        in otherwise normal images less likely to trigger a false positive.
        """
        max_pos = np.unravel_index(np.argmax(patch_score_map), patch_score_map.shape)
        s_star = patch_score_map[max_pos]
        m_star = patch_features_map[max_pos].reshape(1, -1).astype("float32")
        n_star_idx = int(neighbor_idx_map[max_pos][0])

        n_star = self.memory_bank[n_star_idx].numpy().reshape(1, -1).astype("float32")
        k = min(self.config.reweight_k, self.index.ntotal)
        _, neighbor_ids = self.index.search(n_star, k)

        candidates = self.memory_bank[neighbor_ids[0]].numpy().astype("float32")
        dists_to_mstar = np.linalg.norm(candidates - m_star, axis=1)

        # Softmax-style stabilization (subtract max before exp) to avoid overflow,
        # the ratio is unchanged since it cancels out.
        stabilizer = max(s_star, dists_to_mstar.max())
        weight = 1 - (
            np.exp(s_star - stabilizer) / np.sum(np.exp(dists_to_mstar - stabilizer) + 1e-9)
        )
        return float(weight * s_star)

    def save(self):
        out_dir = self.config.model_dir()
        faiss.write_index(self.index, str(out_dir / "memory_bank.faiss"))
        torch.save(self.memory_bank, out_dir / "memory_bank.pt")
        with open(out_dir / "config.json", "w") as f:
            json.dump(
                {
                    "category": self.config.category,
                    "backbone": self.config.backbone,
                    "layers": list(self.config.layers),
                    "image_size": self.config.image_size,
                    "crop_size": self.config.crop_size,
                    "pool_kernel": self.config.pool_kernel,
                    "coreset_ratio": self.config.coreset_ratio,
                    "projection_dim": self.config.projection_dim,
                    "num_neighbors": self.config.num_neighbors,
                    "reweight_k": self.config.reweight_k,
                    "feature_map_size": self.feature_map_size,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, category: str, device: str = "cpu"):
        from src.config import MODELS_ROOT

        model_dir = MODELS_ROOT / category
        with open(model_dir / "config.json") as f:
            saved = json.load(f)

        feature_map_size = saved.pop("feature_map_size", None)
        config = PatchCoreConfig(device=device, **saved)
        model = cls(config)
        model.index = faiss.read_index(str(model_dir / "memory_bank.faiss"))
        model.memory_bank = torch.load(model_dir / "memory_bank.pt", weights_only=True)
        model.feature_map_size = tuple(feature_map_size) if feature_map_size else None
        return model
