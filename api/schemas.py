from pydantic import BaseModel


class PredictionResponse(BaseModel):
    category: str
    image_score: float
    threshold: float
    is_anomaly: bool
    heatmap_base64: str | None = None


class HealthResponse(BaseModel):
    status: str
    loaded_categories: list[str]
