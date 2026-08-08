import torch
from tqdm import tqdm

from src.feature_extractor import PatchFeatureExtractor


@torch.no_grad()
def extract_all_patch_features(extractor: PatchFeatureExtractor, dataloader, device: str = "cpu") -> torch.Tensor:
    """Runs the feature extractor over every training image and returns a
    single (N_patches, C) tensor of all patch features."""
    extractor = extractor.to(device)
    all_features = []
    for batch in tqdm(dataloader, desc="Extracting patch features"):
        images = batch["image"].to(device)
        feature_map = extractor(images)
        all_features.append(extractor.flatten_patches(feature_map).cpu())
    return torch.cat(all_features, dim=0)


class GreedyCoresetSampler:
    """Greedy k-center coreset selection (Sener & Savarese, 2018), as used by
    PatchCore to shrink the memory bank while preserving feature-space coverage.

    A random projection reduces dimensionality before the O(N * k) pairwise
    distance updates, since the greedy search only needs approximate distances
    to pick well-spread points, not exact ones.
    """

    def __init__(self, ratio: float = 0.01, projection_dim: int = 128, device: str = "cpu", seed: int = 42):
        self.ratio = ratio
        self.projection_dim = projection_dim
        self.device = device
        self.seed = seed

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] <= self.projection_dim:
            return features
        generator = torch.Generator().manual_seed(self.seed)
        proj = torch.randn(features.shape[1], self.projection_dim, generator=generator)
        proj = proj / proj.norm(dim=0, keepdim=True)
        return features @ proj

    def sample(self, features: torch.Tensor) -> torch.Tensor:
        n_samples = max(1, int(len(features) * self.ratio))
        features = features.to(self.device)
        projected = self._project(features).to(self.device)

        n = projected.shape[0]
        selected_idx = [torch.randint(0, n, (1,), generator=torch.Generator().manual_seed(self.seed)).item()]
        min_distances = torch.cdist(projected, projected[selected_idx]).squeeze(1)

        for _ in tqdm(range(n_samples - 1), desc="Greedy coreset selection"):
            next_idx = int(torch.argmax(min_distances).item())
            selected_idx.append(next_idx)
            new_distances = torch.cdist(projected, projected[[next_idx]]).squeeze(1)
            min_distances = torch.minimum(min_distances, new_distances)

        return features[selected_idx]
