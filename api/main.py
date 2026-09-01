"""FastAPI backend.

    Frontend -> FastAPI -> Predictor -> Model

The model is loaded exactly once, at startup, and reused for every request.
No model logic lives in the route handlers themselves -- they only validate
the request, call the predictor, and shape the HTTP response.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from skin_disease.inference import InvalidImageError, SkinDiseasePredictor
from skin_disease.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

MEDICAL_DISCLAIMER = (
    "This tool provides an AI-generated classification based on the uploaded "
    "image. It is intended for research and decision-support purposes only "
    "and is not a substitute for evaluation by a qualified healthcare "
    "professional. Model confidence does not represent medical certainty. "
    "If you have a concerning or changing skin lesion or symptom, seek "
    "professional medical advice."
)

_predictor: Optional[SkinDiseasePredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    checkpoint_path = os.environ.get("MODEL_CHECKPOINT_PATH", "models/best_model.pt")
    hf_repo_id = os.environ.get("HF_MODEL_REPO_ID")
    try:
        if hf_repo_id:
            _predictor = SkinDiseasePredictor(hf_repo_id=hf_repo_id)
        else:
            _predictor = SkinDiseasePredictor(checkpoint_path=checkpoint_path)
        logger.info("Model loaded successfully (version=%s)", _predictor.model_version)
    except Exception:
        # Do not crash the whole app: /health will honestly report "not loaded"
        # and /predict will return 503, instead of the process failing to boot.
        logger.exception("Failed to load model at startup.")
        _predictor = None
    yield
    _predictor = None


app = FastAPI(title="Skin Disease Classifier API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3_predictions: list
    severity: str
    model_version: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    num_classes: Optional[int] = None
    device: Optional[str] = None


def get_predictor() -> SkinDiseasePredictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return _predictor


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _predictor is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=_predictor.model_version,
        num_classes=len(_predictor.class_names),
        device=str(_predictor.device),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    gradcam: bool = Query(False, description="Reserved; use /predict/gradcam for the image overlay."),
) -> PredictionResponse:
    predictor = get_predictor()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use JPEG, PNG, or WebP.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")

    try:
        result = predictor.predict(raw)
    except InvalidImageError as exc:
        logger.warning("Rejected invalid upload: %s", exc)
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.") from exc
    except Exception:
        logger.exception("Inference failed.")
        raise HTTPException(status_code=500, detail="Inference failed.") from None

    logger.info("Prediction completed: class=%s confidence=%.4f", result["predicted_class"], result["confidence"])
    return PredictionResponse(**result, disclaimer=MEDICAL_DISCLAIMER)


@app.post("/predict/gradcam")
async def predict_gradcam(file: UploadFile = File(...)) -> dict:
    """Returns the same prediction plus a base64-encoded Grad-CAM overlay PNG."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    predictor = get_predictor()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use JPEG, PNG, or WebP.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large.")

    try:
        result = predictor.predict(raw)
        cam_result = predictor.generate_gradcam(raw)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.") from exc
    except Exception:
        logger.exception("Grad-CAM generation failed.")
        raise HTTPException(status_code=500, detail="Inference failed.") from None

    overlay_uint8 = (np.clip(cam_result["overlay"], 0, 1) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(overlay_uint8).save(buffer, format="PNG")
    overlay_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return {
        **result,
        "disclaimer": MEDICAL_DISCLAIMER,
        "gradcam_target_class": cam_result["target_class_name"],
        "gradcam_overlay_png_base64": overlay_b64,
    }
