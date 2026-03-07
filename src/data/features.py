"""Feature engineering for energy demand forecasting."""

import pandas as pd
import numpy as np


# Ordered list of feature columns used by models
FEATURE_COLUMNS = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    "is_holiday",
    "lag_1", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_std_24",
    "rolling_mean_168", "rolling_std_168",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features using sin/cos encoding.

    Encodes hour-of-day, day-of-week, month, and day-of-year as
    sin/cos pairs to preserve circular continuity (e.g., hour 23 is
    close to hour 0). Also adds a US federal holiday indicator.
    """
    df = df.copy()

    # Hour of day (0-23)
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    # Day of week (0-6)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Month (1-12)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Day of year (1-366)
    df["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    # US federal holiday indicator
    try:
        from pandas.tseries.holiday import USFederalHolidayCalendar

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df.index.min(), end=df.index.max())
        df["is_holiday"] = df.index.normalize().isin(holidays).astype(float)
    except ImportError:
        df["is_holiday"] = 0.0

    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "consumption_scaled",
) -> pd.DataFrame:
    """Add lag features: t-1 (1 hr), t-24 (1 day), t-168 (1 week)."""
    df = df.copy()
    df["lag_1"] = df[target_col].shift(1)
    df["lag_24"] = df[target_col].shift(24)
    df["lag_168"] = df[target_col].shift(168)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "consumption_scaled",
) -> pd.DataFrame:
    """Add rolling mean and std for 24-hour and 168-hour (1-week) windows."""
    df = df.copy()
    df["rolling_mean_24"] = (
        df[target_col].rolling(window=24, min_periods=1).mean()
    )
    df["rolling_std_24"] = (
        df[target_col].rolling(window=24, min_periods=1).std().fillna(0)
    )
    df["rolling_mean_168"] = (
        df[target_col].rolling(window=168, min_periods=1).mean()
    )
    df["rolling_std_168"] = (
        df[target_col].rolling(window=168, min_periods=1).std().fillna(0)
    )
    return df


def create_features(
    df: pd.DataFrame,
    target_col: str = "consumption_scaled",
) -> pd.DataFrame:
    """Apply all feature engineering steps.

    Pipeline: time features -> lag features -> rolling features -> dropna.
    Rows with NaN (from lag/rolling computation) are dropped.
    """
    df = add_time_features(df)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    df = df.dropna()
    return df
