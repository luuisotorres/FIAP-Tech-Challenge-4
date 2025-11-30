# tests/conftest.py
import pytest
import pandas as pd

@pytest.fixture
def mock_stock_df():
    """
    Creates a deterministic 20-day stock dataframe for testing.
    Patterns:
    - Price rises for 10 days, falls for 10 days.
    - Volume is constant.
    """
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
    
    # Create a simple predictable pattern
    prices = [100 + i for i in range(10)] + [110 - i for i in range(10)]
    
    df = pd.DataFrame({
        "open": prices,
        "high": [p + 1 for p in prices],
        "low": [p - 1 for p in prices],
        "close": prices,
        "volume": [1000.0] * 20
    }, index=dates)
    
    # Ensure datetime column is a column, not index (matching loader output)
    df = df.reset_index().rename(columns={"index": "ts"})
    return df