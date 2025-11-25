from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

from src.services.preprocess import create_sequences
from src.services.model_utils import load_model_and_scaler
from src.services.model_train import LSTMModel, train_model

# =====================================================
# 📘 API Metadata
# =====================================================
description = """
This API provides stock price predictions using a **Long Short-Term Memory (LSTM)** neural network model.

Developed for the **Tech Challenge 4 - Deep Learning and AI Module**,  
implemented by the team:

- Izabelly de Oliveira Menezes  
- Larissa Diniz da Silva  
- Luis Fernando Torres  
- Rafael Dos Santos Callegari  
- Renato Massamitsu Zama Inomata  

The model takes recent stock closing prices as input and returns the **next 5-day forecast**.
"""

app = FastAPI(
    title="LSTM Stock Prediction API",
    description=description,
    version="1.0.0"
)

# =====================================================
# 🧠 Model + Scaler Configuration
# =====================================================
MODEL_PATH = "models/lstm_model.pth"
SCALER_PATH = "models/scaler.save"

# Load model and scaler if files exist
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model, scaler = load_model_and_scaler(LSTMModel, MODEL_PATH, SCALER_PATH)
else:
    model = None
    scaler = None

# Device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of past values used for prediction
SEQ_LENGTH = 75

# Number of days to forecast
N_DAYS_FORECAST = 5


# =====================================================
# 📥 Input Schemas
# =====================================================
class PricesInput(BaseModel):
    prices: list[float]


class TrainData(BaseModel):
    valores: list[float]
    epochs: int = 75


# =====================================================
# 🔮 Prediction Logic
# =====================================================
def predict_n_days(model, scaler, prices, seq_length=SEQ_LENGTH, n_days=N_DAYS_FORECAST):
    """
    Generates multi-step forecasts by recursively feeding model predictions
    back into the next input window.
    """
    model.eval()

    data = np.array(prices).reshape(-1, 1)
    scaled_data = scaler.transform(data)

    # Last sequence used as input window
    last_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).to(device)

    preds = []

    for _ in range(n_days):
        # Predict next scaled value
        with torch.no_grad():
            next_scaled = model(last_seq_tensor).cpu().numpy()

        # Inverse transform to real scale
        next_val = scaler.inverse_transform(next_scaled)[0, 0]
        preds.append(float(next_val))

        # Update sequence by appending prediction
        next_tensor = torch.tensor(
            next_scaled.reshape(1, 1, 1), dtype=torch.float32
        ).to(device)

        last_seq_tensor = torch.cat(
            (last_seq_tensor[:, 1:, :], next_tensor),
            dim=1
        )

    return preds


# =====================================================
# 🌐 ROOT ENDPOINT
# =====================================================
@app.get("/", tags=["Root"])
def root():
    return {"message": "LSTM Prediction API is running!"}


# =====================================================
# 🚀 INFERENCE ENDPOINT
# =====================================================
@app.post("/predict", tags=["Inference"])
async def predict(input_data: PricesInput):

    # Check model availability
    if model is None or scaler is None:
        raise HTTPException(status_code=400, detail="Model has not been trained yet.")

    prices = input_data.prices

    # Ensure enough data points
    if len(prices) < SEQ_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Send at least {SEQ_LENGTH} values for prediction."
        )

    preds = predict_n_days(model, scaler, prices)
    return {"predicted_next_5_days": preds}


# =====================================================
# 🏋️ TRAINING (Placeholder for future implementation)
# =====================================================



# =====================================================
# 🔧 MODEL UPDATE (Placeholder for future implementation)
# =====================================================



# =====================================================
# 📊 MONITORING ENDPOINT
# =====================================================
@app.get("/monitoring", tags=["Monitoring"])
def monitoring():

    # Check model file timestamp
    last_modified_unix = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    return {
        "status": "ok",  # API is running
        "model_loaded": model is not None,  # Model loaded in memory
        "scaler_loaded": scaler is not None,  # Scaler loaded
        "model_file_exists": os.path.exists(MODEL_PATH),  # .pth file exists
        "scaler_file_exists": os.path.exists(SCALER_PATH),  # Scaler file exists
        "model_last_modified_unix": last_modified_unix,  # Unix timestamp
        "model_last_modified_iso": datetime.fromtimestamp(last_modified_unix).isoformat() if last_modified_unix else None,  # ISO timestamp
        "model_size_kb": os.path.getsize(MODEL_PATH) / 1024 if os.path.exists(MODEL_PATH) else None,  # Model size in KB
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad) if model is not None else None,  # Trainable params
        "device_in_use": str(device),  # CPU/GPU
        "cuda_available": torch.cuda.is_available(),  # GPU availability
        "seq_length": SEQ_LENGTH,  # Window size used for predictions
        "forecast_horizon": N_DAYS_FORECAST,  # Forecast length
        "torch_version": torch.__version__,  # PyTorch version
        "numpy_version": np.__version__,  # NumPy version
    }


# =====================================================
# ⚙️ CONFIGURATION ENDPOINT
# =====================================================
@app.get("/config", tags=["Configuration"])
def config():
    # Model last modification timestamp
    last_modified_unix = os.path.getmtime(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    return {
        "env": "dev",  # Execution environment ("dev" or "prod")
        "version": app.version,  # API version
        "model_path": MODEL_PATH,  # Model path
        "scaler_path": SCALER_PATH,  # Scaler path
        "model_last_modified_unix": last_modified_unix,  # Unix timestamp
        "model_last_modified_iso": datetime.fromtimestamp(last_modified_unix).isoformat() if last_modified_unix else None,  # ISO date
        "cache_enabled": True,  # Whether cache is enabled
        "use_fallback_cache": True,  # Use cached model if unavailable
        "expected_params": ["prices"],  # Expected body parameters
        "timeouts": {"request": 10},  # Internal timeout settings
        "logging_level": "INFO",  # Logging level
    }
