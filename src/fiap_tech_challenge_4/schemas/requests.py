from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fiap_tech_challenge_4.config import DataStrategyConfig, ModelParams

# --- Training Schemas ---
class TrainingRequest(BaseModel):
    """Payload to trigger a training job."""
    experiment_name: str = "Stock_Forecaster_API"
    epochs: int = 10
    learning_rate: float = 1e-3
    
    # Reuse existing config models for validation consistency
    data: DataStrategyConfig
    # Use Dict for model params to avoid Pydantic V2 conflict in nested init
    model: Dict[str, Any] = Field(default_factory=lambda: {"hidden_dim": 64}) 

class TrainingResponse(BaseModel):
    message: str
    run_id: str
    status: str

# --- Inference Schemas ---
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
    candles: List[StockCandle] = Field(..., description="List of historical candles.")

class PredictionResponse(BaseModel):
    ticker: str
    predicted_return: float
    predicted_price: float

# --- Promotion Schemas ---
class PromotionResponse(BaseModel):
    status: str
    message: str
    current_run_id: str