"""Baseline models for benchmarking against deep learning approaches.

Includes:
    - SeasonalNaive: Predicts using same hour from the previous week
    - ProphetBaseline: Facebook Prophet with daily/weekly/yearly seasonality
"""

import numpy as np
import pandas as pd


class SeasonalNaive:
    """Seasonal naive baseline: predict using the same hour from one week ago.

    This is a strong baseline for energy demand because consumption patterns
    repeat weekly (weekday vs weekend, same hour of day).

    Args:
        season_length: Number of timesteps in one season (168 = 1 week of hourly data).
    """

    def __init__(self, season_length: int = 168):
        self.season_length = season_length
        self.history = None

    def fit(self, y: np.ndarray) -> "SeasonalNaive":
        """Store historical values for generating forecasts."""
        self.history = y.copy()
        return self

    def predict(self, steps: int = 24) -> np.ndarray:
        """Predict next `steps` hours using values from one season ago."""
        start_idx = len(self.history) - self.season_length
        return self.history[start_idx : start_idx + steps]


class ProphetBaseline:
    """Facebook Prophet baseline with automatic seasonality decomposition.

    Prophet decomposes the time series into trend + yearly/weekly/daily
    seasonality components. Useful as a strong statistical baseline.
    """

    def __init__(self):
        self.model = None

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "consumption_mw",
    ) -> "ProphetBaseline":
        """Fit Prophet model on historical data.

        Args:
            df: DataFrame with datetime index and target column.
            target_col: Name of the column to forecast.
        """
        from prophet import Prophet

        prophet_df = df.reset_index()
        prophet_df = prophet_df.rename(
            columns={prophet_df.columns[0]: "ds", target_col: "y"}
        )
        prophet_df = prophet_df[["ds", "y"]]

        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        self.model.fit(prophet_df)
        return self

    def predict(self, periods: int = 24, freq: str = "h") -> pd.DataFrame:
        """Generate future predictions.

        Args:
            periods: Number of future timesteps to forecast.
            freq: Frequency string for the forecast horizon.

        Returns:
            DataFrame with 'ds' (timestamp) and 'yhat' (prediction) columns.
        """
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast.tail(periods)[["ds", "yhat"]].reset_index(drop=True)
