from pydantic import BaseModel, Field
from typing import List


class MacdConfig(BaseModel):
    fast: int = 12
    slow: int = 26
    signal: int = 9


class VolumeConfig(BaseModel):
    log: bool = True
    sma_windows: List[int] = Field(default_factory=list)


class StrategyConfig(BaseModel):
    ticker: str
    period: str = "2y"
    interval: str = "1d"

    # Feature parameters with defaults
    lags: List[int] = [1, 2, 3]
    rolling_windows: List[int] = [5, 20]

    # Nested configs
    macd: MacdConfig = Field(default_factory=MacdConfig)
    volume: VolumeConfig = Field(default_factory=VolumeConfig)