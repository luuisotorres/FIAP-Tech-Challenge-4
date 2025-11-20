import numpy as np
import pandas as pd
from typing import List


# Constant for division stability to avoid ZeroDivisionError or Inf
EPSILON = 1e-9


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the DataFrame by handling infinite values and dropping NaNs.

    This is the final step in the feature engineering pipeline to ensure 
    the data is ready for model ingestion.

    Args:
        df (pd.DataFrame): The input DataFrame containing calculated features.

    Returns:
        pd.DataFrame: A cleaned DataFrame with no infinite or missing values.
    """
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def add_log_return(df: pd.DataFrame, col: str = "close", output_col: str = "r_close") -> pd.DataFrame:
    """Calculates the logarithmic return of a specific column.

    Log returns are preferred in financial modeling due to time-additivity.
    Formula: ln(P_t) - ln(P_{t-1})

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): The price column to compute returns on. Defaults to "close".
        output_col (str): The name of the new column. Defaults to "r_close".

    Returns:
        pd.DataFrame: DataFrame with the added log return column.
    """
    df[output_col] = np.log(df[col]).diff()
    return df


def add_lags(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    """Adds lagged versions of a specific column to capture autocorrelation.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): The column name to lag (e.g., "r_close").
        lags (List[int]): A list of integer lag steps (e.g., [1, 2, 3]).

    Returns:
        pd.DataFrame: DataFrame with added lag columns (e.g., "r_close_lag_1").
    """
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_momentum(
    df: pd.DataFrame,
    col: str,
    windows: List[int],
    aggs: List[str] = ["mean", "sum"]
) -> pd.DataFrame:
    """Calculates rolling statistics (momentum/trend) on a target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): The column to aggregate (usually log returns "r_close").
        windows (List[int]): List of window sizes (e.g., [5, 20]).
        aggs (List[str]): List of aggregations to perform. Options: "mean", "sum".
                          Note: A rolling sum of log returns is equivalent to 
                          the log of the total return over the period.

    Returns:
        pd.DataFrame: DataFrame with added momentum columns.
    """
    for w in windows:
        if "sum" in aggs:
            df[f"{col}_sum_{w}"] = df[col].rolling(w).sum()
        if "mean" in aggs:
            df[f"{col}_mean_{w}"] = df[col].rolling(w).mean()
    return df


def add_rolling_volatility(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    """Calculates rolling standard deviation (volatility).

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): The column to analyze (usually log returns "r_close").
        windows (List[int]): List of window sizes.

    Returns:
        pd.DataFrame: DataFrame with added volatility columns (e.g., "vol_20").
    """
    for w in windows:
        df[f"vol_{w}"] = df[col].rolling(w).std()
    return df


def add_price_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Adds features describing the intraday price structure (candles).

    Calculates:
        - hl_range: Normalized High-Low range.
        - body: Normalized Open-Close body size.
        - gap: Normalized overnight gap (Open - Prev Close).

    Args:
        df (pd.DataFrame): Input DataFrame. Must contain 'high', 'low', 'open', 'close'.

    Returns:
        pd.DataFrame: DataFrame with 'hl_range', 'body', and 'gap' columns.
    """
    prev_close = df["close"].shift(1)
    # High - Low normalized by previous close
    df["hl_range"] = (df["high"] - df["low"]) / (prev_close + EPSILON)
    # Body (Close - Open) normalized by Open
    df["body"] = (df["close"] - df["open"]) / (df["open"] + EPSILON)
    # Gap (Open - Prev Close) normalized by Prev Close
    df["gap"] = (df["open"] - prev_close) / (prev_close + EPSILON)
    return df


def add_moving_averages(
    df: pd.DataFrame,
    col: str,
    sma_windows: List[int] = [],
    ema_windows: List[int] = []
) -> pd.DataFrame:
    """Adds Simple and Exponential Moving Average ratios.

    The features are normalized by the current price to make them 
    scale-invariant (stationarity friendly). 
    Feature = MA / Price.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column to average (usually "close").
        sma_windows (List[int]): Windows for Simple Moving Averages.
        ema_windows (List[int]): Spans for Exponential Moving Averages.

    Returns:
        pd.DataFrame: DataFrame with added normalized MA columns.
    """
    for w in sma_windows:
        sma = df[col].rolling(w).mean()
        df[f"sma_{w}"] = sma / (df[col] + EPSILON)

    for w in ema_windows:
        ema = df[col].ewm(span=w, adjust=False).mean()
        df[f"ema_{w}"] = ema / (df[col] + EPSILON)

    return df


def add_rsi(df: pd.DataFrame, col: str = "close", window: int = 14) -> pd.DataFrame:
    """Calculates the Relative Strength Index (RSI).

    RSI is normalized here to be between 0 and 1 (divided by 100) to 
    match the scale of other neural network features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Price column. Defaults to "close".
        window (int): Lookback period. Defaults to 14.

    Returns:
        pd.DataFrame: DataFrame with added f"rsi_{window}" column.
    """
    delta = df[col].diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)

    # Use exponential moving average for wilder smoothing (standard RSI)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()

    rs = roll_up / (roll_down + EPSILON)
    rsi = 100 - (100 / (1 + rs))

    df[f"rsi_{window}"] = rsi / 100.0
    return df


def add_macd(
    df: pd.DataFrame,
    col: str = "close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """Calculates Moving Average Convergence Divergence (MACD).

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Price column. Defaults to "close".
        fast (int): Fast EMA span. Defaults to 12.
        slow (int): Slow EMA span. Defaults to 26.
        signal (int): Signal line EMA span. Defaults to 9.

    Returns:
        pd.DataFrame: DataFrame with 'macd' and 'macd_signal' columns.
    """
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()

    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    col: str = "close",
    window: int = 20,
    k: float = 2.0
) -> pd.DataFrame:
    """Calculates Bollinger Band Width.

    Adds a 'bb_width' feature which represents the normalized distance 
    between upper and lower bands.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Price column. Defaults to "close".
        window (int): Rolling window for SMA. Defaults to 20.
        k (float): Number of standard deviations. Defaults to 2.0.

    Returns:
        pd.DataFrame: DataFrame with added 'bb_width' column.
    """
    sma = df[col].rolling(window).mean()
    std = df[col].rolling(window).std()

    upper = sma + k * std
    lower = sma - k * std

    # Bandwidth normalized by SMA
    df["bb_width"] = (upper - lower) / (sma + EPSILON)
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculates Average True Range (ATR).

    ATR is a measure of market volatility.

    Args:
        df (pd.DataFrame): Input DataFrame. Must contain high/low/close.
        window (int): Rolling window. Defaults to 14.

    Returns:
        pd.DataFrame: DataFrame with added f"atr_{window}" column.
    """
    prev_close = df["close"].shift(1)

    # True Range is max of: H-L, |H-PrevClose|, |L-PrevClose|
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs()
        )
    )

    df[f"atr_{window}"] = tr.rolling(window).mean()
    return df


def add_volume_features(
    df: pd.DataFrame,
    log_volume: bool = True,
    sma_windows: List[int] = []
) -> pd.DataFrame:
    """Adds volume-based features.

    Args:
        df (pd.DataFrame): Input DataFrame. Must contain "volume".
        log_volume (bool): If True, adds 'log_vol' = log(1 + volume).
        sma_windows (List[int]): List of windows for Volume Moving Averages.
                                 Calculated as Volume / RollingMean(Volume).

    Returns:
        pd.DataFrame: DataFrame with added volume features.
    """
    if log_volume:
        df["log_vol"] = np.log1p(df["volume"])

    for w in sma_windows:
        vol_sma = df["volume"].rolling(w).mean()
        # Normalize current volume by its moving average
        df[f"vol_sma_{w}"] = df["volume"] / (vol_sma + EPSILON)

    return df


def add_calendar_features(df: pd.DataFrame, date_col: str = "ts") -> pd.DataFrame:
    """Adds calendar-based categorical features (Day of Week).

    Performs One-Hot Encoding for the day of the week (Mon=0 to Fri=4).

    Args:
        df (pd.DataFrame): Input DataFrame.
        date_col (str): The name of the datetime column. Defaults to "ts".

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns (e.g., "dow_0").
    """
    if date_col not in df.columns:
        return df

    # Ensure datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    dow = df[date_col].dt.dayofweek

    # Manual OHE for safety and control over column names
    for day in range(5):  # 0=Monday, 4=Friday
        df[f"dow_{day}"] = (dow == day).astype("int8")

    return df
