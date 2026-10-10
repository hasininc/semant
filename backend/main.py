
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Load the trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run python train.py first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="SemanticShield API",
    description="Text sentiment analysis and robustness testing",
    version="1.0.0",
)

# Allow the local React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


@app.get("/")
def home():
    return {
        "message": "SemanticShield API is running!",
        "model": "TF-IDF + Logistic Regression",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(request: PredictionRequest):
    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty.",
        )

    prediction = int(model.predict([text])[0])
    probabilities = model.predict_proba([text])[0]

    return {
        "text": text,
        "sentiment": "Positive" if prediction == 1 else "Negative",
        "confidence": round(
            float(max(probabilities)) * 100, 2
        ),
        "probabilities": {
            "negative": round(float(probabilities[0]) * 100, 2),
            "positive": round(float(probabilities[1]) * 100, 2),
        },
        "model": "TF-IDF + Logistic Regression",
    }