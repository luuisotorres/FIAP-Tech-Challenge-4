import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock

from fiap_tech_challenge_4.config import DataStrategyConfig, TechnicalsConfig
from fiap_tech_challenge_4.features.pipeline import DataPipeline
from fiap_tech_challenge_4.features.strategies import TrendFollowingStrategy


@pytest.fixture
def mock_stock_df():
    dates = pd.date_range(start="2023-01-01", periods=100)
    df = pd.DataFrame({
        "open": np.random.rand(100) * 100,
        "high": np.random.rand(100) * 100,
        "low": np.random.rand(100) * 100,
        "close": np.linspace(100, 200, 100),
        "volume": np.random.rand(100) * 1000
    }, index=dates)
    df.index.name = "Date"
    return df.reset_index().rename(columns={"index": "ts"})


@patch("fiap_tech_challenge_4.features.pipeline.fetch_data")
@patch("fiap_tech_challenge_4.features.pipeline.mlflow")
def test_pipeline_execution(mock_mlflow, mock_fetch, mock_stock_df):
    mock_fetch.return_value = mock_stock_df

    cfg = DataStrategyConfig(
        ticker="AAPL",
        seq_len=10,
        batch_size=5,
        train_split=0.8,
        technicals=TechnicalsConfig(sma=[5])
    )
    strategy = TrendFollowingStrategy(cfg)
    pipeline = DataPipeline(strategy, cfg)

    loaders = pipeline.run()

    assert "train" in loaders
    assert "val" in loaders

    X_batch, y_batch = next(iter(loaders["train"]))

    # Check batch dimensions (Batch, Seq_Len, Features)
    assert X_batch.shape[0] == 5
    assert X_batch.shape[1] == 10
    assert X_batch.shape[2] >= 3
    assert y_batch.shape == (5, 1)


def test_inverse_transform_logic():
    cfg = DataStrategyConfig(ticker="TEST")
    pipeline = DataPipeline(MagicMock(), cfg)

    # Manually set identity scaler (mean=0, scale=1)
    pipeline.target_scaler.mean_ = np.array([0.0])
    pipeline.target_scaler.scale_ = np.array([1.0])

    # Input: 1% log return repeated twice
    scaled_preds = torch.tensor([[0.01], [0.01]])
    last_price = 100.0

    prices = pipeline.inverse_transform_price(scaled_preds, last_price)

    # Day 1: 100 * exp(0.01) ≈ 101.005
    # Day 2: 101.005 * exp(0.01) ≈ 102.02
    assert len(prices) == 2
    assert prices[0] > 100.0
    assert prices[1] > prices[0]

