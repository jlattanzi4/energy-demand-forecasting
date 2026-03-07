"""FastAPI application for serving energy demand forecasts.

Endpoints:
    GET  /         — Health check (model loaded status)
    POST /predict  — Generate 24-hour energy demand forecast

Run locally:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pathlib import Path

from api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.predict import load_model, load_scaler, predict
from src.models.lstm import EnergyLSTM
from src.models.tcn import EnergyTCN

# Global model and scaler (loaded at startup)
_model = None
_scaler = None
_model_type = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler into memory on startup."""
    global _model, _scaler, _model_type

    model_dir = Path("models")
    scaler_path = model_dir / "scaler.pkl"

    # Prefer LSTM, fall back to TCN
    if (model_dir / "lstm_best.pth").exists():
        model_path = model_dir / "lstm_best.pth"
        _model_type = "lstm"
    elif (model_dir / "tcn_best.pth").exists():
        model_path = model_dir / "tcn_best.pth"
        _model_type = "tcn"
    else:
        model_path = None
        print("WARNING: No trained model found. Train a model first.")

    if model_path:
        print(f"Loading {_model_type} model from {model_path}")
        _model = load_model(str(model_path), model_type=_model_type)
        print("Model loaded successfully")

    if scaler_path.exists():
        _scaler = load_scaler(str(scaler_path))
        print("Scaler loaded successfully")

    yield

    # Cleanup
    _model = None
    _scaler = None
    _model_type = None


app = FastAPI(
    title="Energy Demand Forecasting API",
    description=(
        "Predict hourly energy consumption for PJM East region "
        "using PyTorch LSTM/TCN models trained on 10+ years of historical data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check — reports whether model and scaler are loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        scaler_loaded=_scaler is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    """Generate a 24-hour energy demand forecast.

    Accepts at least 336 hours of recent hourly consumption data and
    returns predicted MW for the next 24 hours.
    """
    if _model is None or _scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train a model first with: python -m src.train",
        )

    try:
        # Build DataFrame from request
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(request.timestamps),
                "consumption_mw": request.consumption_mw,
            }
        )
        df = df.set_index("datetime").sort_index()

        if len(df) < 336:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 336 hourly observations, got {len(df)}",
            )

        # Generate forecast
        predictions = predict(_model, df, _scaler)

        # Build forecast timestamps (next 24 hours after the last observation)
        last_ts = df.index[-1]
        forecast_timestamps = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1),
            periods=24,
            freq="h",
        )

        return PredictionResponse(
            timestamps=[ts.isoformat() for ts in forecast_timestamps],
            predicted_mw=predictions.tolist(),
            model_type=_model_type or "unknown",
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
