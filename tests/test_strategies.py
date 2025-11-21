# tests/test_strategies.py
import pytest
from fiap_tech_challenge_4.config import DataStrategyConfig, TechnicalsConfig, MacdConfig
from fiap_tech_challenge_4.features.strategies import TrendFollowingStrategy, MeanReversionStrategy

def test_trend_strategy_columns(mock_stock_df):
    # Configure
    cfg = DataStrategyConfig(
        ticker="AAPL",
        strategy_type="trend",
        rolling_windows=[5],
        technicals=TechnicalsConfig(sma=[10], macd=MacdConfig())
    )
    
    # Run Strategy
    strat = TrendFollowingStrategy(cfg)
    result = strat.apply_features(mock_stock_df)
    
    # Assert Columns exist
    cols = result.columns
    assert "r_close" in cols       # Base
    assert "sma_10" in cols        # Configured SMA
    assert "macd" in cols          # Configured MACD
    assert "mom_5" in cols         # Configured Momentum
    assert "rsi_14" not in cols    # Should NOT be in Trend Strategy

def test_mean_reversion_strategy_columns(mock_stock_df):
    cfg = DataStrategyConfig(
        ticker="AAPL",
        strategy_type="mean_reversion",
        rolling_windows=[5]
    )
    
    strat = MeanReversionStrategy(cfg)
    result = strat.apply_features(mock_stock_df)
    
    assert "rsi_14" in result.columns
    assert "vol_5" in result.columns
    assert "macd" not in result.columns