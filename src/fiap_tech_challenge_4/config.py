from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class MacdConfig(BaseModel):
    """Configuration for Moving Average Convergence Divergence."""
    fast: int = 12
    slow: int = 26
    signal: int = 9


class TechnicalsConfig(BaseModel):
    """Configuration for technical indicators."""
    sma: List[int] = Field(default_factory=lambda: [20, 50])
    macd: Optional[MacdConfig] = Field(default_factory=MacdConfig)


class DataStrategyConfig(BaseModel):
    """
    Configuration for the Data Processing Strategy.

    Controls how data is fetched, engineered, and scaled.
    """
    ticker: str = 'AAPL'
    period: str = "5y"
    interval: str = "1d"

    # Logic parameters
    strategy_type: Literal["trend", "mean_reversion"] = "trend"
    scaler_type: Literal["standard", "minmax", "robust"] = "standard"
    train_split: float = 0.8
    seq_len: int = 60
    batch_size: int = 32

    # Feature specific settings
    rolling_windows: List[int] = Field(default_factory=lambda: [5, 20])
    technicals: TechnicalsConfig = Field(default_factory=TechnicalsConfig)

    @property
    def min_history_required(self) -> int:
        """
        Calculates how many days of data the client MUST send to avoid NaNs.
        Formula: Sequence Length + Max Lag needed for feature engineering + Buffer.
        """
        # Get max window from rolling stats
        max_rolling = max(self.rolling_windows) if self.rolling_windows else 0
        
        # Get max window from SMAs
        max_sma = max(self.technicals.sma) if self.technicals.sma else 0
        
        # MACD usually requires 26 (slow) + 9 (signal) = ~35 days to settle
        max_macd = 35 if self.technicals.macd else 0
        
        # We need the largest of these lags
        max_lag = max(max_rolling, max_sma, max_macd)
        
        # We add a small buffer (5) for differencing offsets (diff)
        return self.seq_len + max_lag + 5


class ModelParams(BaseModel):
    """Hyperparameters for the LSTM architecture."""
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2


class TrainingConfig(BaseModel):
    """
    Master configuration for a Training Job.

    Combines Data settings and Model settings.
    """
    experiment_name: str = "Stock_Forecaster_v1"
    epochs: int = 10
    learning_rate: float = 1e-3

    # Composition
    data: DataStrategyConfig
    model: ModelParams = Field(default_factory=ModelParams)