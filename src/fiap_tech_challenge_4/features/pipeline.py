import torch
import mlflow
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Any

from fiap_tech_challenge_4.config import DataStrategyConfig
from fiap_tech_challenge_4.features.strategies import DataStrategy
from fiap_tech_challenge_4.data.loader import fetch_data


class DataPipeline:
    """Orchestrates data fetching, feature engineering, scaling, and loader creation."""

    def __init__(self, strategy: DataStrategy, config: DataStrategyConfig):
        self.strategy = strategy
        self.cfg = config
        self.feature_scaler = self._get_scaler(config.scaler_type)
        self.target_scaler = self._get_scaler(config.scaler_type)

    def _get_scaler(self, scaler_type: str) -> Any:
        if scaler_type == "minmax":
            return MinMaxScaler(feature_range=(-1, 1))
        elif scaler_type == "robust":
            return RobustScaler()
        elif scaler_type == "standard":
            return StandardScaler()

        raise ValueError(
            f"Invalid scaler_type: '{scaler_type}'. Must be 'minmax', 'robust', or 'standard'."
        )

    def run(self) -> Dict[str, DataLoader]:
        """Executes the pipeline and returns training and validation dataloaders."""
        with mlflow.start_run(nested=True):
            mlflow.log_params({
                "ticker": self.cfg.ticker,
                "strategy": type(self.strategy).__name__,
                "scaler": self.cfg.scaler_type,
                "seq_len": self.cfg.seq_len,
                "split_ratio": self.cfg.train_split
            })

            raw_df = fetch_data(
                ticker=self.cfg.ticker,
                period=self.cfg.period,
                interval=self.cfg.interval
            )

            processed_df = self.strategy.apply_features(raw_df)
            mlflow.log_metric("num_features_generated", processed_df.shape[1])

            # Drop artifacts from lagging/differencing
            processed_df = processed_df.dropna()

            target_col = "r_close"

            # Drop the target column from potential features
            # Select ONLY numeric columns (floats/ints).
            potential_features = processed_df.drop(
                columns=[target_col], errors="ignore")
            numeric_features = potential_features.select_dtypes(include=[
                                                                np.number])

            feature_cols = numeric_features.columns.tolist()

            # Align features (t) with targets (t+1)
            features = processed_df[feature_cols].values[:-1]
            targets = processed_df[target_col].shift(
                -1).dropna().values.reshape(-1, 1)

            # Ensure strict alignment length
            min_len = min(len(features), len(targets))
            features = features[:min_len]
            targets = targets[:min_len]

            mlflow.log_metric("total_samples", len(features))
            mlflow.log_metric("final_feature_count", features.shape[1])

            # Chronological split
            split_idx = int(len(features) * self.cfg.train_split)
            X_train, X_val = features[:split_idx], features[split_idx:]
            y_train, y_val = targets[:split_idx], targets[split_idx:]

            # Fit scaler only on training set to avoid leakage
            X_train = self.feature_scaler.fit_transform(X_train)
            y_train = self.target_scaler.fit_transform(y_train)

            X_val = self.feature_scaler.transform(X_val)
            y_val = self.target_scaler.transform(y_val)

            return {
                "train": self._create_loader(X_train, y_train, shuffle=True),
                "val": self._create_loader(X_val, y_val, shuffle=False)
            }

    def _create_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        X_seq, y_seq = [], []
        seq_len = self.cfg.seq_len

        for i in range(len(X) - seq_len):
            X_seq.append(X[i: i + seq_len])
            y_seq.append(y[i + seq_len])

        if not X_seq:
            raise ValueError(
                f"Dataset too small ({len(X)}) for sequence length ({seq_len}).")

        X_t = torch.tensor(np.stack(X_seq), dtype=torch.float32)
        y_t = torch.tensor(np.stack(y_seq), dtype=torch.float32)

        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            drop_last=True
        )

    def inverse_transform_price(self, predicted_scaled_returns: torch.Tensor, last_known_price: float) -> np.ndarray:
        """Converts scaled log-return predictions back to absolute price path."""
        if isinstance(predicted_scaled_returns, torch.Tensor):
            preds = predicted_scaled_returns.detach().cpu().numpy()
        else:
            preds = predicted_scaled_returns

        log_returns = self.target_scaler.inverse_transform(preds)
        price_multipliers = np.exp(log_returns).flatten()

        return last_known_price * np.cumprod(price_multipliers)
