from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from fiap_tech_challenge_4.config import DataStrategyConfig, ModelParams


# Training Schemas
class TrainingRequest(BaseModel):
    """Payload to trigger a training job."""
    experiment_name: str = "Stock_Forecaster_API"
    epochs: int = 10
    learning_rate: float = 1e-3

    data: DataStrategyConfig
    model: Dict[str, Any] = Field(default_factory=lambda: {"hidden_dim": 64})

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "experiment_name": "Stock_Forecaster_API",
                "epochs": 5,
                "learning_rate": 0.001,
                "data": {
                    "ticker": "AAPL",
                    "period": "1y",
                    "interval": "1d",
                    "strategy_type": "trend",
                    "scaler_type": "robust",
                    "train_split": 0.8,
                    "seq_len": 30,
                    "batch_size": 32,
                    "rolling_windows": [5, 20],
                    "technicals": {
                        "sma": [20, 50],
                        "macd": {
                            "fast": 12,
                            "slow": 26,
                            "signal": 9
                        }
                    }
                },
                "model": {
                    "hidden_dim": 64,
                    "num_layers": 2,
                    "dropout": 0.2
                }
            }
        }
    )


class TrainingResponse(BaseModel):
    message: str
    run_id: str
    status: str


# Inference Schemas
class StockCandle(BaseModel):
    """Represents a single time-step of OHLCV data."""
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictionRequest(BaseModel):
    """
    Payload for inference. 
    Must contain enough historical candles to satisfy the model's sequence length + lag.
    """
    ticker: str
    candles: List[StockCandle] = Field(...,
                                       description="List of historical candles.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ticker": "AAPL",
                "candles": [
                    {
                        "open": 150.0,
                        "high": 155.0,
                        "low": 149.0,
                        "close": 153.0,
                        "volume": 5000000.0
                    }
                ]
            }
        }
    )


class PredictionResponse(BaseModel):
    ticker: str
    predicted_return: float
    predicted_price: float


# Promotion Schemas
class PromotionResponse(BaseModel):
    status: str
    message: str
    current_run_id: str


# Observability Schemas
class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool


class ModelMetadataResponse(BaseModel):
    """Details about the currently active production model."""
    run_id: str = "unknown"
    experiment_name: str
    strategy_type: str
    seq_len: int
    model_params: Dict[str, Any]