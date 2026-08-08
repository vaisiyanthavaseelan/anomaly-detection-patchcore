import numpy as np
from skimage import measure
from sklearn.metrics import precision_recall_curve, roc_auc_score


def image_auroc(labels: np.ndarray, image_scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, image_scores))


def best_f1_threshold(labels: np.ndarray, image_scores: np.ndarray) -> float:
    """Threshold on the validation/test scores that maximizes F1. Used as the
    default decision boundary served by the API; recalibrate per deployment
    if the acceptable false-positive/false-negative trade-off differs."""
    precision, recall, thresholds = precision_recall_curve(labels, image_scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[best_idx]) if len(thresholds) else float(np.median(image_scores))


def pixel_auroc(gt_masks: np.ndarray, score_maps: np.ndarray) -> float:
    return float(roc_auc_score(gt_masks.flatten() > 0.5, score_maps.flatten()))


def pro_score(gt_masks: np.ndarray, score_maps: np.ndarray, num_thresholds: int = 200, max_fpr: float = 0.3) -> float:
    """Per-Region Overlap score (Bergmann et al., MVTec-AD), integrated up to a
    false-positive rate of `max_fpr`. Unlike pixel AUROC, this weights small
    defect regions the same as large ones, since averaging is done per region
    rather than per pixel.
    """
    thresholds = np.linspace(score_maps.min(), score_maps.max(), num_thresholds)
    normal_mask = gt_masks <= 0.5

    pro_values = []
    fpr_values = []

    for t in thresholds:
        binarized = score_maps >= t

        region_overlaps = []
        for i in range(gt_masks.shape[0]):
            labeled_gt = measure.label(gt_masks[i, 0] > 0.5)
            for region_id in range(1, labeled_gt.max() + 1):
                region = labeled_gt == region_id
                overlap = np.logical_and(region, binarized[i, 0]).sum()
                region_overlaps.append(overlap / region.sum())

        pro = float(np.mean(region_overlaps)) if region_overlaps else 0.0

        fp = np.logical_and(binarized, normal_mask).sum()
        fpr = fp / max(normal_mask.sum(), 1)

        pro_values.append(pro)
        fpr_values.append(fpr)

    fpr_values = np.array(fpr_values)
    pro_values = np.array(pro_values)

    order = np.argsort(fpr_values)
    fpr_values, pro_values = fpr_values[order], pro_values[order]

    mask = fpr_values <= max_fpr
    if mask.sum() < 2:
        return 0.0

    fpr_capped, pro_capped = fpr_values[mask], pro_values[mask]
    auc = float(np.trapz(pro_capped, fpr_capped) / max_fpr)
    return auc


def evaluate_category(labels, image_scores, gt_masks, score_maps):
    return {
        "image_auroc": image_auroc(labels, image_scores),
        "pixel_auroc": pixel_auroc(gt_masks, score_maps),
        "pro_score": pro_score(gt_masks, score_maps),
    }
