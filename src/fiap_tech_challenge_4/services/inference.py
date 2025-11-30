import pandas as pd
import torch
import numpy as np

from fiap_tech_challenge_4.schemas.requests import PredictionRequest, PredictionResponse
from fiap_tech_challenge_4.core.state import production_model
from fiap_tech_challenge_4.features.strategies import TrendFollowingStrategy, MeanReversionStrategy


def predict_next_step(payload: PredictionRequest) -> PredictionResponse:
    if not production_model.is_loaded:
        raise RuntimeError(
            "No production model loaded. Please train or promote a model first.")

    # Convert Payload to DataFrame
    df = pd.DataFrame([c.model_dump() for c in payload.candles])

    # Re-hydrate the Strategy using the PRODUCTION Config
    prod_cfg = production_model.config.data

    if prod_cfg.strategy_type == "trend":
        strategy = TrendFollowingStrategy(prod_cfg)
    elif prod_cfg.strategy_type == "mean_reversion":
        strategy = MeanReversionStrategy(prod_cfg)
    else:
        raise ValueError(
            f"Unknown strategy type in production config: {prod_cfg.strategy_type}"
        )

    # Feature Engineering
    processed_df = strategy.apply_features(df)

    # Handle Feature Lag
    # The client must send at least (seq_len + max_rolling_window) candles.
    processed_df = processed_df.dropna()

    # Prepare Features for Inference
    target_col = "r_close"
    exclude_cols = {"ts", "date", target_col}

    # Get the exact columns the model expects (from the fitted scaler)
    # This relies on the columns being in the same order as training
    numeric_df = processed_df.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_df.columns if c not in exclude_cols]

    features = processed_df[feature_cols].values

    # Check if we have enough data for one sequence
    seq_len = prod_cfg.seq_len
    if len(features) < seq_len:
        raise ValueError(
            f"Not enough data after feature engineering. "
            f"Needed {seq_len}, got {len(features)}. "
            f"Send more historical candles."
        )

    # Take the last sequence
    last_sequence = features[-seq_len:]

    # Scale using PRODUCTION Scaler
    scaled_seq = production_model.feature_scaler.transform(last_sequence)

    # Predict
    # Shape: (1, seq_len, input_dim)
    input_tensor = torch.tensor(scaled_seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        # Output: (1, 1)
        pred_scaled = production_model.model(input_tensor)

    # Inverse Transform
    # We convert the scaled return back to a raw log return
    pred_log_return = production_model.target_scaler.inverse_transform(
        pred_scaled.numpy()
    ).item()

    # Calculate Price
    # Price_t+1 = Price_t * exp(log_return)
    last_close = payload.candles[-1].close
    pred_price = last_close * np.exp(pred_log_return)

    return PredictionResponse(
        ticker=payload.ticker,
        predicted_return=pred_log_return,
        predicted_price=pred_price
    )
