> [!WARNING]
> **Archived, November 2026.** This project is not maintained, and its headline result overstates the models. The "58% lower MAE" below is measured against a same-hour-last-week baseline, which is a weak yardstick for a day-ahead forecast: the same hour yesterday, or a gradient-boosted model on the same features, would be much harder to beat. That comparison was never run ([#1](https://github.com/jlattanzi4/energy-demand-forecasting/issues/1)), so treat the results table as unverified. The code is kept as it was.

# Energy Demand Forecasting with PyTorch

End-to-end time series forecasting system that predicts hourly energy consumption for the PJM East regional transmission network. Built with PyTorch LSTM and Temporal Convolutional Network (TCN) architectures, benchmarked against Prophet and Seasonal Naive baselines, served via FastAPI, containerized with Docker, and deployable on AWS.

## Business Context

Utilities need accurate short-term load forecasts to optimize power generation scheduling. Over-forecasting wastes fuel; under-forecasting risks blackouts. A 1% improvement in forecast accuracy can save millions annually for a large utility. This system provides 24-hour-ahead hourly consumption predictions for the PJM East region.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Pipeline                         │
│  Kaggle CSV → Preprocessing → Feature Engineering        │
│  (145K hourly obs)  (scaling)   (17 features)            │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │   LSTM   │  │   TCN    │  │ Baselines│
   │ (2-layer │  │ (Dilated │  │ (Naive,  │
   │  128 hid)│  │  Causal) │  │ Prophet) │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │     Evaluation Module       │
         │  MAE · RMSE · MAPE · Plots  │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │     FastAPI + Docker        │
         │   POST /predict → 24h MW    │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────┐
         │     AWS Lambda / EC2        │
         │   API Gateway endpoint      │
         └────────────────────────────┘
```

## Dataset

**PJM Hourly Energy Consumption** — [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

| Attribute   | Detail                                              |
|-------------|-----------------------------------------------------|
| Region      | PJM East (PJME)                                     |
| Records     | ~145,000 hourly observations                         |
| Time Span   | 2002–2018                                            |
| Granularity  | Hourly, megawatts (MW)                              |
| License     | Public domain                                        |

## Models

| Model | Description | Key Design Choice |
|-------|-------------|-------------------|
| **Seasonal Naive** | Same hour from previous week | Strong baseline for weekly patterns |
| **Prophet** | Additive model + seasonality | Automatic trend + seasonality decomposition |
| **LSTM** | 2-layer, 128 hidden, dropout 0.2 | Captures long-range temporal dependencies |
| **TCN** | 4 blocks, 64 channels, kernel 3 | Faster training, dilated receptive field |

### Feature Engineering (17 features)
- **Cyclical time**: hour, day-of-week, month, day-of-year (sin/cos pairs)
- **Holiday**: US federal holiday indicator
- **Lags**: t-1, t-24, t-168 (1hr, 1day, 1week)
- **Rolling stats**: 24hr and 168hr mean and std

## Results

| Model | MAE (MW) | RMSE (MW) | MAPE (%) |
|-------|----------|-----------|----------|
| Seasonal Naive | 3526.31 | 4811.11 | 11.05 |
| Prophet | 3699.82 | 4744.04 | 11.95 |
| **LSTM** | **1470.04** | **2034.86** | **4.65** |
| **TCN** | 1499.49 | 2037.28 | 4.79 |

LSTM achieved the best performance across all metrics, reducing MAE by 58% compared to the Seasonal Naive baseline.

## Quick Start

### 1. Setup

```bash
git clone <repo-url>
cd energy-demand-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data

Download `PJME_hourly.csv` from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) and place it in `data/`.

### 3. Train Models

```bash
# Train both LSTM and TCN
python -m src.train --data-path data/PJME_hourly.csv --model both --epochs 50

# Train only LSTM
python -m src.train --model lstm --epochs 50

# Train only TCN
python -m src.train --model tcn --epochs 50
```

### 4. Run API Locally

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Test the health endpoint:
```bash
curl http://localhost:8000/
```

Generate a forecast:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### 5. Docker

```bash
docker build -t energy-forecast .
docker run -p 8000:8000 energy-forecast
```

### 6. Deploy to AWS

```bash
cd deploy
sam build
sam deploy --guided
```

## Project Structure

```
energy-demand-forecasting/
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   └── 02_modeling.ipynb         # Model training & comparison
├── src/
│   ├── data/
│   │   ├── preprocess.py         # Load, clean, split, scale
│   │   └── features.py           # Feature engineering pipeline
│   ├── models/
│   │   ├── lstm.py               # PyTorch LSTM architecture
│   │   ├── tcn.py                # Temporal Convolutional Network
│   │   └── baselines.py          # Seasonal Naive + Prophet
│   ├── train.py                  # Training loop + early stopping
│   ├── evaluate.py               # Metrics + visualization
│   └── predict.py                # Inference module
├── api/
│   ├── app.py                    # FastAPI application
│   └── schemas.py                # Pydantic request/response models
├── deploy/
│   ├── lambda_handler.py         # AWS Lambda wrapper (Mangum)
│   └── template.yaml             # SAM/CloudFormation template
├── models/                       # Saved checkpoints (.pth, .pkl)
└── data/                         # Raw CSV data (gitignored)
```

## Technology Stack

| Layer | Tools |
|-------|-------|
| Data Processing | pandas, numpy, scikit-learn |
| Deep Learning | PyTorch (LSTM, TCN) |
| Statistical Models | Prophet |
| API | FastAPI + Uvicorn |
| Containerization | Docker |
| Deployment | AWS Lambda + API Gateway |
| Experiment Tracking | MLflow (optional) |

## Future Improvements

- Add weather covariates (temperature, humidity) via Open-Meteo API
- Implement Temporal Fusion Transformer for multi-horizon forecasting
- Set up automated retraining with Airflow or AWS Step Functions
- Add data drift detection and model monitoring
- A/B testing infrastructure for model deployment
- Streamlit dashboard for interactive predictions
