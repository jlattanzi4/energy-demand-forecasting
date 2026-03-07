"""Data preprocessing for energy demand forecasting."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle


def load_data(data_path: str = "data/PJME_hourly.csv") -> pd.DataFrame:
    """Load PJM East hourly energy consumption data.

    Args:
        data_path: Path to the PJME CSV file from Kaggle.

    Returns:
        DataFrame with datetime index and consumption_mw column,
        sorted chronologically with gaps filled via linear interpolation.
    """
    df = pd.read_csv(data_path)
    df.columns = ["datetime", "consumption_mw"]
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.set_index("datetime")
    df = df[~df.index.duplicated(keep="first")]

    # Reindex to continuous hourly frequency, interpolate gaps
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    df = df.reindex(full_range)
    df.index.name = "datetime"
    df.columns = ["consumption_mw"]
    df["consumption_mw"] = df["consumption_mw"].interpolate(method="linear")

    return df


def temporal_split(
    df: pd.DataFrame,
    val_start: str = "2016-01-01",
    test_start: str = "2017-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data using temporal cutoffs (no random shuffling).

    Default split:
        Train: 2002 - end of 2015
        Val:   2016
        Test:  2017 - 2018

    Args:
        df: Full dataset with datetime index.
        val_start: Start date for validation set.
        test_start: Start date for test set.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train = df[df.index < val_start].copy()
    val = df[(df.index >= val_start) & (df.index < test_start)].copy()
    test = df[df.index >= test_start].copy()
    return train, val, test


def fit_scaler(
    train_df: pd.DataFrame,
    save_path: str = "models/scaler.pkl",
) -> StandardScaler:
    """Fit StandardScaler on training data only and save to disk.

    Args:
        train_df: Training DataFrame with consumption_mw column.
        save_path: Where to save the fitted scaler.

    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[["consumption_mw"]])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(scaler, f)

    return scaler


def transform_data(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply scaler transformation to consumption data.

    Adds a 'consumption_scaled' column to the DataFrame.
    """
    df = df.copy()
    df["consumption_scaled"] = scaler.transform(df[["consumption_mw"]])
    return df
