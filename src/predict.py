"""Inference module for generating energy demand predictions.

Loads a trained model and scaler, processes recent historical data,
and generates a 24-hour forecast in MW.

Usage:
    from src.predict import load_model, load_scaler, predict
    model = load_model("models/lstm_best.pth", model_type="lstm")
    scaler = load_scaler("models/scaler.pkl")
    forecast_mw = predict(model, recent_df, scaler)
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from src.data.features import create_features, FEATURE_COLUMNS
from src.models.lstm import EnergyLSTM
from src.models.tcn import EnergyTCN


def load_model(
    model_path: str,
    model_type: str = "lstm",
    input_size: int = 17,
    device: str | None = None,
) -> torch.nn.Module:
    """Load a trained model from a checkpoint file.

    Args:
        model_path: Path to the .pth checkpoint.
        model_type: "lstm" or "tcn".
        input_size: Number of input features (must match training config).
        device: Compute device (auto-detected if None).

    Returns:
        Model in eval mode on the specified device.
    """
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    if model_type == "lstm":
        model = EnergyLSTM(input_size=input_size)
    elif model_type == "tcn":
        model = EnergyTCN(input_size=input_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def load_scaler(scaler_path: str = "models/scaler.pkl"):
    """Load a fitted StandardScaler from disk."""
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def predict(
    model: torch.nn.Module,
    recent_data: pd.DataFrame,
    scaler,
    device: str | None = None,
    window_size: int = 168,
) -> np.ndarray:
    """Generate a 24-hour forecast from recent historical data.

    Args:
        model: Trained PyTorch model (EnergyLSTM or EnergyTCN).
        recent_data: DataFrame with datetime index and 'consumption_mw' column.
            Must have enough rows to produce at least `window_size` rows
            after feature engineering (recommend 336+ hours / 2 weeks).
        scaler: Fitted StandardScaler for inverse transformation.
        device: Compute device (auto-detected if None).
        window_size: Input window size matching training config.

    Returns:
        Array of 24 predicted values in MW (inverse-transformed).
    """
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    # Scale the data
    df = recent_data.copy()
    df["consumption_scaled"] = scaler.transform(df[["consumption_mw"]])

    # Engineer features
    df = create_features(df)

    if len(df) < window_size:
        raise ValueError(
            f"Need at least {window_size} rows after feature engineering, got {len(df)}. "
            f"Provide more historical data (recommend 336+ hours)."
        )

    # Extract the most recent window
    window = df[FEATURE_COLUMNS].values[-window_size:]
    x = torch.FloatTensor(window).unsqueeze(0).to(device)

    # Generate forecast
    with torch.no_grad():
        pred_scaled = model(x).cpu().numpy().flatten()

    # Inverse-transform back to MW
    pred_mw = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    return pred_mw
