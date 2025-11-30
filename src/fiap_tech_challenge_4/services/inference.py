import pandas as pd
import torch
import numpy as np

from fiap_tech_challenge_4.schemas.requests import PredictionRequest, PredictionResponse
from fiap_tech_challenge_4.core.state import production_model
from fiap_tech_challenge_4.features.strategies import TrendFollowingStrategy, MeanReversionStrategy


def predict_next_step(payload: PredictionRequest) -> PredictionResponse:
    """
    Orchestrates the inference pipeline: Data -> Features -> Scale -> Model -> Inverse.
    """
    if not production_model.is_loaded:
        raise RuntimeError(
            "No production model loaded. Train or promote a model first.")

    # Convert Payload to DataFrame
    df = pd.DataFrame([c.model_dump() for c in payload.candles])

    # Re-hydrate Strategy from Production Config
    prod_cfg = production_model.config.data

    if prod_cfg.strategy_type == "trend":
        strategy = TrendFollowingStrategy(prod_cfg)
    elif prod_cfg.strategy_type == "mean_reversion":
        strategy = MeanReversionStrategy(prod_cfg)
    else:
        raise ValueError(f"Unknown strategy: {prod_cfg.strategy_type}")

    # Feature Engineering
    processed_df = strategy.apply_features(df)
    processed_df = processed_df.dropna()

    # Feature Selection
    target_col = "r_close"
    exclude_cols = {"ts", "date", target_col}

    numeric_df = processed_df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c not in exclude_cols]

    features = processed_df[feature_cols].values

    # Data Sufficiency Check
    seq_len = prod_cfg.seq_len
    if len(features) < seq_len:
        raise ValueError(
            f"Not enough valid data after feature engineering. "
            f"Model needs {seq_len} steps, but only {len(features)} are valid. "
            f"Please send at least {prod_cfg.min_history_required} candles."
        )

    # Take the most recent sequence
    last_sequence = features[-seq_len:]

    # Scale & Predict
    scaled_seq = production_model.feature_scaler.transform(last_sequence)
    input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred_scaled = production_model.model(input_tensor)

    # Inverse Transform
    pred_log_return = production_model.target_scaler.inverse_transform(
        pred_scaled.numpy()
    ).item()

    last_close = payload.candles[-1].close
    pred_price = last_close * np.exp(pred_log_return)

    return PredictionResponse(
        ticker=payload.ticker,
        predicted_return=pred_log_return,
        predicted_price=pred_price
    )
