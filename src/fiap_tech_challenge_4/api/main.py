from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import torch

# from fiap_tech_challenge_4.api.preprocess import create_sequences
from fiap_tech_challenge_4.api.model_utils import load_model_and_scaler
from fiap_tech_challenge_4.api.model_train import LSTMModel

app = FastAPI(title="LSTM Stock Prediction API")

# Model and scaler paths
MODEL_PATH = "models/lstm_model.pth"
SCALER_PATH = "models/scaler.save"

# Load model and scaler
model, scaler = load_model_and_scaler(LSTMModel, MODEL_PATH, SCALER_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

SEQ_LENGTH = 75  # same sequence length used during training
N_DAYS_FORECAST = 5  # next 5 days forecast


# Input model
class PricesInput(BaseModel):
    prices: list[float]  # list of closing prices


# N-step prediction function
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
        preds.append(float(next_pred))  # convert to native float

        # Update sequence with the new prediction
        next_pred_scaled_tensor = torch.tensor(
            next_pred_scaled.reshape(1, 1, 1), dtype=torch.float32
        ).to(device)
        last_seq_tensor = torch.cat((last_seq_tensor[:, 1:, :], next_pred_scaled_tensor), dim=1)

    return preds


@app.post("/predict")
async def predict(input_data: PricesInput):
    prices = input_data.prices
    if len(prices) < SEQ_LENGTH:
        return {"error": f"At least {SEQ_LENGTH} prices are required for prediction."}

    preds = predict_n_days(model, scaler, prices, seq_length=SEQ_LENGTH, n_days=N_DAYS_FORECAST)
    return {"predictions_next_5_days": preds}
