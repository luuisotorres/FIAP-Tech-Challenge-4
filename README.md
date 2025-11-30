# FIAP Tech Challenge 4 - Stock Forecaster API

A production-grade MLOps application for predicting stock prices using Long Short-Term Memory (LSTM) networks. This project implements a complete end-to-end pipeline including data ingestion, feature engineering, model training, experiment tracking, and serving via a high-performance Async API.

## 🚀 Overview

This application is designed with a **Service-Oriented Architecture (SOA)** pattern to decouple the Machine Learning "Engine" from the API "Interface".

**Key Features:**
* **Deep Learning:** Custom LSTM architecture with MLP head for time-series regression.
* **Modular Pipeline:** Robust feature engineering (RSI, MACD, SMA) using Pydantic configuration.
* **Experiment Tracking:** Full integration with **MLflow (DagsHub)** for logging metrics, parameters, and artifacts.
* **Async API:** FastAPI implementation with non-blocking training jobs using BackgroundTasks and Global Locks.
* **Model Registry:** "Hot-swap" production models without restarting the server using the `/promote` endpoint.
* **Dockerized:** Ready for deployment with full observability support.

---

## 📂 Project Structure

The project uses the `src` layout managed by `uv`.

```text
stock-forecaster/
├── .env                       # Environment variables (Credentials)
├── docker-compose.yml         # Container orchestration
├── Dockerfile                 # Image definition
├── pyproject.toml             # Dependencies (uv)
├── scripts/                   # Utility scripts (Smoke tests, data generation)
├── tests/                     # Pytest suite
└── src/
    └── fiap_tech_challenge_4/
        ├── config.py          # Pydantic Configurations (Data, Model, Training)
        ├── main.py            # App entry point
        │
        ├── api/               # Interface Layer
        │   ├── app.py         # FastAPI App & Lifespan logic
        │   └── routes.py      # Endpoints (/train, /predict, /promote)
        │
        ├── core/              # State Management
        │   └── state.py       # Singleton ModelArtifacts (In-Memory State)
        │
        ├── data/              # Data Layer
        │   └── loader.py      # yfinance Ingestion
        │
        ├── features/          # Feature Engineering Layer
        │   ├── library.py     # Stateless Math Functions
        │   ├── strategies.py  # Strategy Patterns (Trend/MeanReversion)
        │   └── pipeline.py    # Scaling, Splitting & Tensor Creation
        │
        ├── modeling/          # ML Core Layer
        │   ├── lstm.py              # PyTorch LSTM Architecture
        │   ├── lightning_module.py  # Training Loop Wrapper
        │   └── trainer.py           # Training Orchestrator
        │
        ├── schemas/           # API Contracts (DTOs)
        │   └── requests.py    # Request/Response models
        │
        └── services/          # Business Logic Layer
            ├── inference.py   # Prediction Logic
            ├── training.py    # Training Job Logic
            └── promotion.py   # Model Registry & Hot-Swap Logic
```

---

## 🛠️ Setup & Installation

### Prerequisites
* Python 3.10+
* [uv](https://github.com/astral-sh/uv) (Fast Python package installer)
* Docker (Optional, for container run)

### 1. Clone and Sync
```bash
git clone https://github.com/luuisotorres/FIAP-Tech-Challenge-4.git
cd FIAP-Tech-Challenge-4
uv sync
```

### 2. Configure Environment
Create a `.env` file in the root directory. You need DagsHub credentials for experiment tracking.

```ini
# .env
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<your-repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your-username>
MLFLOW_TRACKING_PASSWORD=<your-token>

# Required for Artifact Downloads (DagsHub S3 Proxy)
MLFLOW_S3_ENDPOINT_URL=https://dagshub.com/api/v1/repo-buckets/s3
AWS_ACCESS_KEY_ID=<your-username>
AWS_SECRET_ACCESS_KEY=<your-token>
```

---

## 🏃 Usage

### Running Locally (Developer Mode)
Start the API with hot-reloading enabled.

```bash
uv run uvicorn fiap_tech_challenge_4.api.app:app --reload --host 0.0.0.0 --port 8000
```

Access the **Swagger UI** at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Running with Docker (Production Mode)
Build and run the entire stack.

```bash
docker compose up --build
```

---

## 📡 API Endpoints

### 1. Train a Model (`POST /v1/train`)
Triggers an asynchronous training job. The API returns `202 Accepted` immediately, and training continues in the background. Metrics are logged to DagsHub.

**Payload Example:**
```json
{
  "experiment_name": "AAPL_Trend_V1",
  "epochs": 15,
  "learning_rate": 0.001,
  "data": {
    "ticker": "AAPL",
    "strategy_type": "trend",
    "scaler_type": "robust",
    "seq_len": 60
  },
  "model": {
    "hidden_dim": 64,
    "num_layers": 2
  }
}
```

### 2. Predict Prices (`POST /v1/predict`)
Returns the forecasted price for the next day (t+1). Requires a list of historical candles equal to `seq_len` + feature lag requirements.

**Note:** You can generate a valid mock payload using the script:
```bash
uv run scripts/generate_mock_payload.py
```

### 3. Promote Model (`POST /v1/model/promote`)
Downloads specific artifacts from a DagsHub MLflow Run and hot-swaps the active model in memory without restarting the server.

**Query Param:** `run_id` (The hash string from MLflow).

---

## 🧪 Testing

The project maintains high test coverage using `pytest` and `unittest.mock` to simulate external services (DagsHub, yfinance).

Run the full suite:
```bash
uv run pytest tests/
```

---

## 👥 Authors
Developed for FIAP - Tech Challenge 4.

* Izabelly de Oliveira Menezes
* Larissa Diniz da Silva
* Luis Fernando Torres
* Rafael Dos Santos Callegari
* Renato Massamitsu Zama Inomata