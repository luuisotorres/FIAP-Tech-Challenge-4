# tests/test_loader.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from fiap_tech_challenge_4.data.loader import fetch_data


@pytest.fixture
def mock_yfinance_data():
    """Simulates the raw dataframe returned by yfinance download."""
    dates = pd.date_range(start="2023-01-01", periods=5)
    df = pd.DataFrame({
        "Open": [100.0] * 5,
        "High": [105.0] * 5,
        "Low": [95.0] * 5,
        "Close": [100.0] * 5,
        "Volume": [1000] * 5
    }, index=dates)

    df.index.name = "Date"
    return df


@patch("fiap_tech_challenge_4.data.loader.yf.download")
def test_fetch_data_success(mock_download, mock_yfinance_data):
    # Setup the mock to return our fixture
    mock_download.return_value = mock_yfinance_data

    # Run our function
    df = fetch_data("AAPL", period="5d")

    # Assertions
    assert not df.empty
    # Check column cleaning (lowercase)
    assert "close" in df.columns
    assert "Close" not in df.columns
    # Check index reset
    assert "ts" in df.columns


@patch("fiap_tech_challenge_4.data.loader.yf.download")
def test_fetch_data_empty_error(mock_download):
    # Simulate empty response (e.g., bad ticker)
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError, match="No data found"):
        fetch_data("INVALID_TICKER")


@patch("fiap_tech_challenge_4.data.loader.yf.download")
def test_missing_column_fill(mock_download):
    # Simulate a response missing 'Volume' (common in some indices)
    incomplete_df = pd.DataFrame({
        "Open": [100], "High": [100], "Low": [100], "Close": [100]
        # No Volume
    }, index=pd.to_datetime(["2023-01-01"]))

    incomplete_df.index.name = "Date"

    mock_download.return_value = incomplete_df

    df = fetch_data("SPX")

    # It should have created the column and filled with 0.0
    assert "volume" in df.columns
    assert df["volume"].iloc[0] == 0.0