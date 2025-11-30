import pytest
import pandas as pd
import numpy as np
from fiap_tech_challenge_4.features import features as F


def test_clean_data():
    # Create dirty data
    df = pd.DataFrame({"a": [1, np.inf, np.nan, 4]})
    clean = F.clean_data(df)

    # Inf and NaN should be removed
    assert len(clean) == 2
    assert clean.iloc[0]["a"] == 1.0
    assert clean.iloc[1]["a"] == 4.0


def test_log_return(mock_stock_df):
    df = F.add_log_return(mock_stock_df, col="close", output_col="r_close")

    # Check calculation: log(101) - log(100) approx 0.00995
    expected = np.log(101) - np.log(100)
    assert "r_close" in df.columns
    assert df["r_close"].iloc[1] == pytest.approx(expected, abs=1e-5)


def test_rsi_calculation(mock_stock_df):
    # RSI requires a window. With our mock data (10 up, 10 down),
    # RSI should be high in the first half, low in the second.
    df = F.add_rsi(mock_stock_df, window=5)

    assert "rsi_5" in df.columns

    # RSI is normalized 0-1 in our library
    rsi_values = df["rsi_5"].dropna()
    assert (rsi_values >= 0).all() and (rsi_values <= 1).all()


def test_moving_averages_stationarity(mock_stock_df):
    # Our feature is SMA / Price (Stationary)
    df = F.add_moving_averages(mock_stock_df, "close", sma_windows=[5])

    # Value should be close to 1.0 (since SMA is close to Price)
    # Not 100.0 (which would be the raw price)
    val = df["sma_5"].iloc[-1]
    assert 0.9 < val < 1.1
