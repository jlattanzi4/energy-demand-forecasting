"""Pydantic request/response schemas for the energy forecasting API."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for energy demand prediction.

    Provide at least 336 hours (2 weeks) of recent hourly consumption data.
    The extra history beyond the 168-hour window is needed because feature
    engineering (lag/rolling features) consumes the first ~168 rows.
    """

    timestamps: list[str] = Field(
        ...,
        description="ISO 8601 datetime strings for each hourly observation",
        min_length=336,
    )
    consumption_mw: list[float] = Field(
        ...,
        description="Hourly energy consumption values in megawatts (MW)",
        min_length=336,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "timestamps": [
                        "2018-01-01T00:00:00",
                        "2018-01-01T01:00:00",
                    ],
                    "consumption_mw": [30000.0, 29500.0],
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response body containing a 24-hour energy demand forecast."""

    timestamps: list[str] = Field(
        ...,
        description="ISO 8601 datetime strings for each forecasted hour",
    )
    predicted_mw: list[float] = Field(
        ...,
        description="Predicted energy consumption in megawatts (MW)",
    )
    model_type: str = Field(
        ...,
        description="Model architecture used for prediction (lstm or tcn)",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    scaler_loaded: bool
