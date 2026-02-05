from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


app = FastAPI(title="mlops_starter", version="0.1.0")


class PredictRequest(BaseModel):
    features: List[float]


class PredictResponse(BaseModel):
    prediction: float
    model_version: str


@app.get("/")
def root():
    return {"message": "mlops_starter API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # Dummy model: sum of features (placeholder)
    pred = float(sum(payload.features))
    return {"prediction": pred, "model_version": "dummy-0.1.0"}
