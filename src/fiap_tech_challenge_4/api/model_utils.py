import os
import torch
import joblib


def save_model(model, model_path: str, scaler, scaler_path: str):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Modelo salvo em {model_path} e scaler salvo em {scaler_path}")


def load_model_and_scaler(model_class, model_path: str, scaler_path: str):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler
