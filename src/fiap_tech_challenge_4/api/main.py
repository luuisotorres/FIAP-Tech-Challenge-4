from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch
from fiap_tech_challenge_4.api.preprocess import create_sequences
from fiap_tech_challenge_4.api.model_utils import load_model_and_scaler
from fiap_tech_challenge_4.api.model_train import LSTMModel

# =====================================================
# 📘 API Metadata for Swagger UI
# =====================================================
description = """
This API provides stock price predictions using a **Long Short-Term Memory (LSTM)** neural network model.

Developed for the **Tech Challenge - Deep Learning and AI Module**,  
demonstrating the complete pipeline of a predictive LSTM model for time series forecasting.

The model takes recent stock closing prices as input and returns the **next 5-day forecast**.
"""

app = FastAPI(
    title="LSTM Stock Prediction API",
    description=description,
    version="1.0.0",
    contact={
        "name": "Tech Challenge 4 Group",
        "email": "group.techchallenge4@fiap.com"
    },
    license_info={
    "name": "Developed by: Izabelly de Oliveira Menezes, Larissa Diniz da Silva, "
            "Luis Fernando Torres, Rafael Dos Santos Callegari, Renato Massamitsu Zama Inomata"
    }
)

# =====================================================
# ✅ Root Endpoint
# =====================================================
@app.get("/", summary="Root endpoint", tags=["Health Check"])
def root():
    return {
        "message": "Welcome to the LSTM Stock Prediction API!"
    }


# =====================================================
# 🧠 Model and Scaler Loading
# =====================================================
MODEL_PATH = "models/lstm_model.pth"
SCALER_PATH = "models/scaler.save"

model, scaler = load_model_and_scaler(LSTMModel, MODEL_PATH, SCALER_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

SEQ_LENGTH = 75         # same sequence length used during training
N_DAYS_FORECAST = 5     # forecast horizon (next 5 days)

# =====================================================
# 📥 Input Schema
# =====================================================
class PricesInput(BaseModel):
    prices: list[float]  # list of closing prices


# =====================================================
# 🔮 Multi-step Prediction Function
# =====================================================
def predict_n_days(model, scaler, prices, seq_length=SEQ_LENGTH, n_days=N_DAYS_FORECAST):
    model.eval()
    data = np.array(prices).reshape(-1, 1)
    scaled_data = scaler.transform(data)

    last_seq = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).to(device)

    preds = []

    for _ in range(n_days):
        with torch.no_grad():
            next_pred_scaled = model(last_seq_tensor).cpu().numpy()
        next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
        preds.append(float(next_pred))  # ensure native Python float

        # Update the sequence with the new prediction
        next_pred_scaled_tensor = torch.tensor(
            next_pred_scaled.reshape(1, 1, 1), dtype=torch.float32
        ).to(device)
        last_seq_tensor = torch.cat((last_seq_tensor[:, 1:, :], next_pred_scaled_tensor), dim=1)

    return preds


# =====================================================
# 🚀 API Endpoint
# =====================================================
@app.post(
    "/predict",
    summary="Predict next 5 stock closing prices",
    response_description="List of predicted stock prices for the next 5 days",
)
async def predict(input_data: PricesInput):
    """
    This endpoint predicts the **next 5 stock closing prices** based on the provided historical prices.
    
    - **prices**: List of recent stock closing prices  
    - **returns**: A list containing predicted prices for the next 5 days
    """
    prices = input_data.prices
    if len(prices) < SEQ_LENGTH:
        return {"error": f"At least {SEQ_LENGTH} prices are required for prediction."}

    preds = predict_n_days(model, scaler, prices, seq_length=SEQ_LENGTH, n_days=N_DAYS_FORECAST)
    return {"predictions_next_5_days": preds}


