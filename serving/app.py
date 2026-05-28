from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from skin_cancer.inference.predict import predict_image
from serving.schemas import PredictionResponse

CONFIG_PATH = os.getenv("MODEL_CONFIG", "configs/train_config.yaml")
CHECKPOINT_PATH = os.getenv("MODEL_CHECKPOINT", "models/best_model.pth")

app = FastAPI(title="Skin Cancer Classification API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not Path(CHECKPOINT_PATH).exists():
        raise HTTPException(status_code=500, detail=f"Model checkpoint not found: {CHECKPOINT_PATH}")

    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_image(tmp_path, CHECKPOINT_PATH, CONFIG_PATH)
        return PredictionResponse(predictions=result["predictions"])
    finally:
        Path(tmp_path).unlink(missing_ok=True)
