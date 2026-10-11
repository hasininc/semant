import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import predict as run_prediction, get_available_models

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
ALL_METRICS_PATH = MODELS_DIR / "all_clean_metrics.json"
ROBUSTNESS_PATH = MODELS_DIR / "robustness_results.json"


# ---------------- FASTAPI APP ----------------

app = FastAPI(
    title="SemanticShield API",
    description="Text sentiment analysis and robustness evaluation across traditional and transformer models",
    version="2.0.0",
)


# ---------------- CORS CONFIGURATION ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8443",
        "http://127.0.0.1:8443",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- REQUEST MODELS ----------------

class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    model: Optional[str] = Field(default="tfidf", description="Model ID: 'tfidf', 'distilbert', 'distilbert_robust'")


# ---------------- ENDPOINTS ----------------

@app.get("/")
def home():
    return {
        "message": "SemanticShield API is running!",
        "version": "2.0.0",
        "supported_models": get_available_models(),
    }


@app.get("/health")
def health():
    models_status = get_available_models()
    any_loaded = any(m["available"] for m in models_status.values())
    return {
        "status": "healthy",
        "models_available": any_loaded,
        "models": models_status,
        "active_default": "tfidf",
    }


@app.get("/models")
def list_models():
    """Lists available model architectures and checkpoint readiness."""
    return {
        "models": get_available_models()
    }


@app.get("/metrics")
def get_metrics(model: Optional[str] = Query(default=None)):
    """Returns evaluation metrics.
    
    If ?model=all, returns clean metrics across all models.
    Otherwise returns specific model metrics or baseline metrics.
    """
    if model == "all" and ALL_METRICS_PATH.exists():
        with open(ALL_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    if model == "distilbert":
        dist_path = MODELS_DIR / "distilbert_metrics.json"
        if dist_path.exists():
            with open(dist_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="DistilBERT metrics not found. Train model B first.")

    if model == "distilbert_robust":
        robust_path = MODELS_DIR / "distilbert_robust_metrics.json"
        if robust_path.exists():
            with open(robust_path, "r", encoding="utf-8") as f:
                return json.load(f)
        raise HTTPException(status_code=404, detail="DistilBERT Robust metrics not found. Train model C first.")

    # Default baseline metrics
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    raise HTTPException(
        status_code=404,
        detail="Metrics not found. Run python train.py first.",
    )


@app.get("/robustness")
def get_robustness_results():
    """Returns measured robustness evaluation results."""
    if not ROBUSTNESS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Robustness results not found. Run python evaluate_robustness.py first.",
        )

    with open(ROBUSTNESS_PATH, "r", encoding="utf-8") as file:
        results = json.load(file)

    return {
        "results": results
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    """Executes sentiment prediction with specified model."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    target_model = request.model or "tfidf"
    # Normalize model ID aliases if frontend sends descriptive names
    target_lower = target_model.lower()
    if "robust" in target_lower:
        model_id = "distilbert_robust"
    elif "distilbert" in target_lower or "transformer" in target_lower:
        model_id = "distilbert"
    else:
        model_id = "tfidf"

    try:
        result = run_prediction(text=text, model_id=model_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
