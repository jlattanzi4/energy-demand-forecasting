"""Evaluation metrics and visualization for model comparison.

Metrics:
    - MAE (Mean Absolute Error): Primary metric, interpretable in MW
    - RMSE (Root Mean Squared Error): Penalizes large forecast errors
    - MAPE (Mean Absolute Percentage Error): Percentage-based for stakeholder reporting

Usage:
    python -m src.evaluate --data-path data/PJME_hourly.csv
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in original units (MW)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error — penalizes large errors."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%). Ignores zero actuals."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Model evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_pytorch_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler=None,
    device: str | None = None,
) -> dict:
    """Evaluate a PyTorch model on the test DataLoader.

    Returns dict with mae, rmse, mape, predictions, and actuals
    (all in original MW scale if scaler is provided).
    """
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    model = model.to(device)
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Inverse-transform back to MW
    if scaler is not None:
        y_pred_flat = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_true_flat = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    else:
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()

    return {
        "mae": mae(y_true_flat, y_pred_flat),
        "rmse": rmse(y_true_flat, y_pred_flat),
        "mape": mape(y_true_flat, y_pred_flat),
        "predictions": y_pred_flat,
        "actuals": y_true_flat,
    }


def evaluate_seasonal_naive(
    test_data: np.ndarray,
    season_length: int = 168,
    horizon: int = 24,
) -> dict:
    """Evaluate seasonal naive baseline with rolling predictions."""
    preds, actuals = [], []

    for i in range(season_length, len(test_data) - horizon + 1, horizon):
        pred = test_data[i - season_length : i - season_length + horizon]
        actual = test_data[i : i + horizon]
        preds.extend(pred)
        actuals.extend(actual)

    y_pred = np.array(preds)
    y_true = np.array(actuals)

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "predictions": y_pred,
        "actuals": y_true,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_comparison_table(results: dict[str, dict]):
    """Print a formatted comparison table of all model results."""
    print(f"\n{'Model':<20} {'MAE (MW)':>12} {'RMSE (MW)':>12} {'MAPE (%)':>12}")
    print("-" * 58)
    for name, metrics in results.items():
        print(
            f"{name:<20} {metrics['mae']:>12.2f} {metrics['rmse']:>12.2f} {metrics['mape']:>12.2f}"
        )
    print("-" * 58)


def plot_predictions(
    results: dict[str, dict],
    sample_size: int = 168,
    save_path: str | None = None,
):
    """Plot actual vs predicted for a sample period across all models."""
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models), sharex=True)

    if n_models == 1:
        axes = [axes]

    for ax, (name, metrics) in zip(axes, results.items()):
        actuals = metrics["actuals"][:sample_size]
        preds = metrics["predictions"][:sample_size]

        ax.plot(actuals, label="Actual", color="#2196F3", linewidth=1.5)
        ax.plot(preds, label="Predicted", color="#FF5722", linewidth=1.5, alpha=0.8)
        ax.set_title(
            f"{name} — MAE: {metrics['mae']:.2f} MW | RMSE: {metrics['rmse']:.2f} MW"
        )
        ax.set_ylabel("Energy (MW)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Hour")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_training_history(
    history: dict,
    model_name: str = "Model",
    save_path: str | None = None,
):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history["train_loss"], label="Train Loss", color="#2196F3")
    ax.plot(history["val_loss"], label="Val Loss", color="#FF5722")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"{model_name} — Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
