import yfinance as yf
import pandas as pd


def fetch_data(
    ticker: str, 
    period: str = "2y", 
    interval: str = "1d", 
    adjusted: bool = True
) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance and standardizes the schema.

    Args:
        ticker (str): The stock symbol (e.g., 'AAPL').
        period (str): Data period to download (e.g., '1d', '5d', '1mo', '1y', 'max').
        interval (str): Data interval (e.g., '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo').
        adjusted (bool): If True, uses 'Adj Close' as 'Close'.

    Returns:
        pd.DataFrame: A cleaned DataFrame with columns: [ts, open, high, low, close, volume].
    
    Raises:
        ValueError: If the downloaded data is empty.
    """
    
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=adjusted,
        multi_level_index=False, 
        progress=False 
    )

    if df.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}' (Period: {period}, Interval: {interval})."
        )

    df = df.reset_index()

    df.columns = df.columns.str.lower().str.strip()
    
    date_col = "date" if "date" in df.columns else "datetime"
    if date_col not in df.columns:
        raise KeyError(f"Could not identify date column. Found: {df.columns.tolist()}")

    df["ts"] = pd.to_datetime(df[date_col])

    required_cols = ["ts", "open", "high", "low", "close", "volume"]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing columns {missing} for {ticker}. Filling with 0 or dropping.")
        for col in missing:
            df[col] = 0.0
    
    final_cols = [c for c in required_cols if c in df.columns]
    df = df[final_cols]

    df = df.sort_values("ts").drop_duplicates("ts")
    df = df.dropna()
    
    return df.reset_index(drop=True)