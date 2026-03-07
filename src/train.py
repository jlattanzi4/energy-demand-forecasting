"""Training pipeline for energy demand forecasting models.

Provides:
    - EnergyDataset: PyTorch Dataset for sliding-window time series
    - create_dataloaders: Build train/val/test DataLoaders
    - train_model: Training loop with early stopping, LR scheduling, gradient clipping
    - prepare_data: Full data preparation pipeline (load -> split -> scale -> features)

Usage:
    python -m src.train --data-path data/PJME_hourly.csv --model both --epochs 50
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse

from src.data.preprocess import load_data, temporal_split, fit_scaler, transform_data
from src.data.features import create_features, FEATURE_COLUMNS
from src.models.lstm import EnergyLSTM
from src.models.tcn import EnergyTCN


class EnergyDataset(Dataset):
    """PyTorch Dataset for sliding-window time series forecasting.

    Creates (input_window, target_horizon) pairs by sliding over the data.

    Args:
        features: Array of shape (n_timesteps, n_features).
        targets: Array of shape (n_timesteps,) with scaled consumption values.
        window_size: Number of historical timesteps per input sample.
        forecast_horizon: Number of future timesteps to predict.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int = 168,
        forecast_horizon: int = 24,
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.n_samples = len(features) - window_size - forecast_horizon + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size : idx + self.window_size + self.forecast_horizon]
        return x, y


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "consumption_scaled",
    window_size: int = 168,
    forecast_horizon: int = 24,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders from preprocessed DataFrames."""
    datasets = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        features = df[feature_cols].values
        targets = df[target_col].values
        datasets[name] = EnergyDataset(features, targets, window_size, forecast_horizon)

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 7,
    model_name: str = "model",
    save_dir: str = "models",
    device: str | None = None,
) -> tuple[nn.Module, dict]:
    """Train a PyTorch model with early stopping and LR scheduling.

    Args:
        model: PyTorch model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        epochs: Maximum training epochs.
        lr: Initial learning rate for Adam optimizer.
        patience: Early stopping patience (epochs without improvement).
        model_name: Name prefix for saved artifacts.
        save_dir: Directory to save model checkpoints and history.
        device: Compute device (auto-detected if None).

    Returns:
        Tuple of (best_model, training_history_dict).
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True,
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"Training {model_name} on {device}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'LR':>10}")
    print("-" * 44)

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"{epoch + 1:>6d} {avg_train_loss:>12.6f} {avg_val_loss:>12.6f} {current_lr:>10.6f}"
        )

        scheduler.step(avg_val_loss)

        # --- Early stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "model_name": model_name,
                },
                save_path / f"{model_name}_best.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Save training history
    with open(save_path / f"{model_name}_history.json", "w") as f:
        json.dump(history, f)

    # Load best model
    checkpoint = torch.load(
        save_path / f"{model_name}_best.pth", weights_only=False, map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nBest val loss: {best_val_loss:.6f} at epoch {checkpoint['epoch'] + 1}")

    return model, history


def prepare_data(data_path: str = "data/PJME_hourly.csv"):
    """Run the full data preparation pipeline.

    Steps: load -> temporal split -> fit scaler -> transform -> feature engineering.

    Returns:
        Tuple of (train_df, val_df, test_df, scaler).
    """
    print("Loading data...")
    df = load_data(data_path)
    print(f"  Loaded {len(df):,} records ({df.index.min()} to {df.index.max()})")

    print("Splitting data (temporal)...")
    train_df, val_df, test_df = temporal_split(df)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    print("Fitting scaler on training data...")
    scaler = fit_scaler(train_df)

    train_df = transform_data(train_df, scaler)
    val_df = transform_data(val_df, scaler)
    test_df = transform_data(test_df, scaler)

    print("Engineering features...")
    train_df = create_features(train_df)
    val_df = create_features(val_df)
    test_df = create_features(test_df)

    print(f"  Features: {len(FEATURE_COLUMNS)} columns")
    print(
        f"  Post-features — Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
    )

    return train_df, val_df, test_df, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train energy forecasting models")
    parser.add_argument("--data-path", type=str, default="data/PJME_hourly.csv")
    parser.add_argument("--model", type=str, default="both", choices=["lstm", "tcn", "both"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=168)
    parser.add_argument("--forecast-horizon", type=int, default=24)
    args = parser.parse_args()

    # Prepare data
    train_df, val_df, test_df, scaler = prepare_data(args.data_path)

    input_size = len(FEATURE_COLUMNS)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df,
        val_df,
        test_df,
        feature_cols=FEATURE_COLUMNS,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
    )

    # Build models to train
    models_to_train = []
    if args.model in ("lstm", "both"):
        models_to_train.append((
            "lstm",
            EnergyLSTM(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                forecast_horizon=args.forecast_horizon,
            ),
        ))
    if args.model in ("tcn", "both"):
        models_to_train.append((
            "tcn",
            EnergyTCN(
                input_size=input_size,
                num_channels=[64, 64, 64, 64],
                kernel_size=3,
                dropout=0.2,
                forecast_horizon=args.forecast_horizon,
            ),
        ))

    # Train each model
    for name, model in models_to_train:
        print(f"\n{'=' * 60}")
        print(f"  Training {name.upper()}")
        print(f"{'=' * 60}")
        train_model(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs,
            lr=args.lr,
            model_name=name,
        )
