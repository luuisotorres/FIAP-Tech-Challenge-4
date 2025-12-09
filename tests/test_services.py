import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from fiap_tech_challenge_4.services.inference import predict_next_step
from fiap_tech_challenge_4.schemas.requests import PredictionRequest, StockCandle
from fiap_tech_challenge_4.config import (
    TrainingConfig,
    DataStrategyConfig, 
    TechnicalsConfig
)


@pytest.fixture
def mock_payload():
    # Create 15 candles (enough for seq_len=10 + lag=5)
    candles = [
        StockCandle(open=100+i, high=105+i, low=95+i, close=100+i, volume=1000)
        for i in range(15)
    ]
    return PredictionRequest(ticker="TEST", candles=candles)


@pytest.fixture
def mock_production_state():
    # Create a mock of the Singleton 'production_model'
    mock_state = MagicMock()
    mock_state.is_loaded = True

    # Config matching the data needs
    mock_state.config = TrainingConfig(
        data=DataStrategyConfig(
            ticker="TEST",
            strategy_type="trend",  # Uses SMA, requires lag
            seq_len=5,  # Small sequence for testing
            rolling_windows=[5],
            technicals=TechnicalsConfig(sma=[5], macd=None)
        ),
        model={}
    )

    # Mock Scalers
    mock_feature_scaler = MagicMock()
    # Transform returns a dummy array of shape (1, seq_len, features)
    mock_feature_scaler.transform.return_value = np.zeros((5, 5))
    mock_state.feature_scaler = mock_feature_scaler

    mock_target_scaler = MagicMock()
    # Inverse transform returns 0.01 (1% return)
    mock_target_scaler.inverse_transform.return_value = np.array([[0.01]])
    mock_state.target_scaler = mock_target_scaler

    # Mock Model
    # Forward pass returns dummy tensor
    mock_state.model.return_value = torch.tensor([[0.5]])

    return mock_state


@patch("fiap_tech_challenge_4.services.inference.production_model")
def test_predict_next_step_success(mock_singleton, mock_payload, mock_production_state):
    # Inject our complex mock into the singleton import
    mock_singleton.is_loaded = mock_production_state.is_loaded
    mock_singleton.config = mock_production_state.config
    mock_singleton.feature_scaler = mock_production_state.feature_scaler
    mock_singleton.target_scaler = mock_production_state.target_scaler
    mock_singleton.model = mock_production_state.model

    # Execute Service
    response = predict_next_step(mock_payload)

    # Validation
    assert response.ticker == "TEST"
    # We mocked inverse_transform to return 0.01
    assert response.predicted_return == 0.01
    # Last close was 100 + 14 = 114.
    # Price = 114 * exp(0.01) ≈ 115.14
    assert response.predicted_price > 114.0


@patch("fiap_tech_challenge_4.services.inference.production_model")
def test_predict_fails_not_loaded(mock_singleton, mock_payload):
    mock_singleton.is_loaded = False

    with pytest.raises(RuntimeError, match="No production model loaded"):
        predict_next_step(mock_payload)


@patch("fiap_tech_challenge_4.services.inference.production_model")
def test_predict_fails_insufficient_data(mock_singleton, mock_production_state):
    # Setup state
    mock_singleton.is_loaded = True
    mock_singleton.config = mock_production_state.config  # seq_len=5

    # Payload with only 1 candle
    short_payload = PredictionRequest(
        ticker="TEST",
        candles=[StockCandle(open=100, high=105, low=95,
                             close=100, volume=1000)]
    )

    with pytest.raises(ValueError, match="Not enough valid data"):
        predict_next_step(short_payload)
