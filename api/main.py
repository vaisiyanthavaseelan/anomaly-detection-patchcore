import base64
import io
import json

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from api.schemas import HealthResponse, PredictionResponse
from src.config import MODELS_ROOT
from src.dataset import build_transform
from src.patchcore import PatchCore

app = FastAPI(title="MTU Bauteil-Anomalieerkennung", description="PatchCore-basierte Defekterkennung fuer metal_nut und screw")

MODELS: dict[str, PatchCore] = {}
THRESHOLDS: dict[str, float] = {}


@app.on_event("startup")
def load_models():
    if not MODELS_ROOT.exists():
        return
    for category_dir in MODELS_ROOT.iterdir():
        if not (category_dir / "memory_bank.faiss").exists():
            continue
        category = category_dir.name
        MODELS[category] = PatchCore.load(category)

        threshold_path = category_dir / "threshold.json"
        THRESHOLDS[category] = json.load(open(threshold_path))["threshold"] if threshold_path.exists() else 0.0


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", loaded_categories=list(MODELS.keys()))


@app.get("/categories")
def categories():
    return {"categories": list(MODELS.keys())}


def _score_map_to_base64(score_map: np.ndarray) -> str:
    normalized = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-9)
    heatmap = (normalized * 255).astype("uint8")
    img = Image.fromarray(heatmap).convert("L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.post("/predict/{category}", response_model=PredictionResponse)
async def predict(category: str, file: UploadFile = File(...), return_heatmap: bool = True):
    if category not in MODELS:
        raise HTTPException(status_code=404, detail=f"Kein trainiertes Modell fuer Kategorie '{category}'. Verfuegbar: {list(MODELS.keys())}")

    model = MODELS[category]
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Ungueltige Bilddatei.")

    transform = build_transform(model.config.image_size, model.config.crop_size, mask=False)
    image_t = transform(image).unsqueeze(0)

    image_scores, score_maps = model.predict(image_t)
    score = float(image_scores[0])
    threshold = THRESHOLDS.get(category, 0.0)

    return PredictionResponse(
        category=category,
        image_score=score,
        threshold=threshold,
        is_anomaly=score > threshold,
        heatmap_base64=_score_map_to_base64(score_maps[0]) if return_heatmap else None,
    )
