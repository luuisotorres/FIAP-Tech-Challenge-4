from pydantic import BaseModel, Field
from typing import List, Optional
from fiap_tech_challenge_4.config import DataStrategyConfig, ModelParams


# Training Schemas
class TrainingRequest(BaseModel):
    """Payload to trigger a training job."""
    experiment_name: str = "Stock_Forecaster_API"
    epochs: int = 10
    learning_rate: float = 1e-3

    # Reuse existing config models for nested validation
    data: DataStrategyConfig
    model: Optional[ModelParams] = Field(default_factory=ModelParams)


class TrainingResponse(BaseModel):
    message: str
    run_id: str
    status: str


# Inference Schemas
class StockCandle(BaseModel):
    """Represents a single time-step of data required for prediction."""
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictionRequest(BaseModel):
    """
    Payload for inference. 
    Must contain enough history (seq_len) to make a prediction.
    """
    ticker: str
    candles: List[StockCandle] = Field(...,
                                       description="List of historical candles (OHLCV).")


class PredictionResponse(BaseModel):
    ticker: str
    predicted_return: float
    predicted_price: float
