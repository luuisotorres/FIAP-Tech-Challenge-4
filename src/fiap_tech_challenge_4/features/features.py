import numpy as np
import pandas as pd
from typing import List

# Constant for division stability
EPSILON = 1e-9


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by handling infinite values and dropping NaNs.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def add_log_return(df: pd.DataFrame, col: str = "close", output_col: str = "r_close") -> pd.DataFrame:
    """
    Calculates the logarithmic return of a specific column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): The price column.
        output_col (str): The name of the new column.

    Returns:
        pd.DataFrame: DataFrame with added log return column.
    """
    df[output_col] = np.log(df[col]).diff()
    return df


def add_moving_averages(df: pd.DataFrame, col: str, sma_windows: List[int]) -> pd.DataFrame:
    """
    Adds normalized Simple Moving Averages (SMA / Price).

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column to average.
        sma_windows (List[int]): Window sizes.

    Returns:
        pd.DataFrame: DataFrame with added sma columns.
    """
    for w in sma_windows:
        sma = df[col].rolling(w).mean()
        # Normalize by current price for stationarity
        df[f"sma_{w}"] = sma / (df[col] + EPSILON)
    return df


def add_rolling_momentum(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    """
    Calculates rolling mean of returns (Momentum).

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column to analyze (usually returns).
        windows (List[int]): Window sizes.

    Returns:
        pd.DataFrame: DataFrame with added momentum columns.
    """
    for w in windows:
        df[f"mom_{w}"] = df[col].rolling(w).mean()
    return df


def add_rolling_volatility(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    """
    Calculates rolling standard deviation.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column to analyze.
        windows (List[int]): Window sizes.

    Returns:
        pd.DataFrame: DataFrame with added volatility columns.
    """
    for w in windows:
        df[f"vol_{w}"] = df[col].rolling(w).std()
    return df


def add_rsi(df: pd.DataFrame, col: str = "close", window: int = 14) -> pd.DataFrame:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Price column.
        window (int): Lookback period.

    Returns:
        pd.DataFrame: DataFrame with added rsi column.
    """
    delta = df[col].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()

    rs = roll_up / (roll_down + EPSILON)
    rsi = 100 - (100 / (1 + rs))

    # Normalize to 0-1 range for Neural Net
    df[f"rsi_{window}"] = rsi / 100.0
    return df


def add_macd(df: pd.DataFrame, fast: int, slow: int, signal: int, col: str = "close") -> pd.DataFrame:
    """
    Calculates MACD and Signal line.

    Args:
        df (pd.DataFrame): Input DataFrame.
        fast (int): Fast EMA span.
        slow (int): Slow EMA span.
        signal (int): Signal line span.
        col (str): Price column.

    Returns:
        pd.DataFrame: DataFrame with 'macd' and 'macd_signal' columns.
    """
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()

    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    return df
