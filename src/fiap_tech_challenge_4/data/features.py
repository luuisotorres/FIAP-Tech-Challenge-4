import pandas as pd
import numpy as np


def log_return(x: pd.Series) -> pd.Series:
    return np.log(x).diff()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    # --- Base log return (training target source) ---
    df["r_close"] = log_return(df["close"])

    # --- Lagged returns (short-term autocorrelation) ---
    for L in cfg["features"].get("lags", []):
        df[f"r_lag_{L}"] = df["r_close"].shift(L)

    # --- Rolling returns: momentum/trend in log-domain ---
    rr_cfg = cfg["features"].get("rolling_returns", {"windows": [5, 20], "agg": ["sum", "mean"]})
    for w in rr_cfg.get("windows", []):
        if "sum" in rr_cfg.get("agg", []):
            df[f"r_sum_{w}"] = df["r_close"].rolling(w).sum()  # ≈ log(close_t/close_{t-w})
        if "mean" in rr_cfg.get("agg", []):
            df[f"r_mean_{w}"] = df["r_close"].rolling(w).mean()

    # --- Rolling volatility: std of returns over windows ---
    for w in cfg["features"].get("rolling_volatility", []):
        df[f"vol_{w}"] = df["r_close"].rolling(w).std()

    # --- Range-based price features (intraday structure) ---
    if cfg["features"].get("range_features", False):
        df["hl_range"] = (df["high"] - df["low"]) / df["close"].shift(1)
        df["body"] = (df["close"] - df["open"]) / df["open"]
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # --- Technical indicators ---
    tech = cfg["features"].get("technicals", {})
    # SMA/EMA ratios (MA normalized by current close to keep scales stable)
    for w in tech.get("sma", []):
        df[f"sma_{w}"] = df["close"].rolling(w).mean() / df["close"]
    for w in tech.get("ema", []):
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean() / df["close"]
    for w in tech.get("rsi", []):
        df[f"rsi_{w}"] = rsi(df["close"], w) / 100.0

    # MACD
    macd_cfg = tech.get("macd")
    if macd_cfg:
        fast = int(macd_cfg.get("fast", 12))
        slow = int(macd_cfg.get("slow", 26))
        sig = int(macd_cfg.get("signal", 9))
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=sig, adjust=False).mean()

    # Bollinger Bands
    bb_cfg = tech.get("bollinger")
    if bb_cfg:
        w = int(bb_cfg.get("window", 20))
        k = float(bb_cfg.get("k", 2))
        sma = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        upper = sma + k * std
        lower = sma - k * std
        df["bb_width"] = (upper - lower) / (sma + 1e-9)

    # ATR
    atr_cfg = tech.get("atr")
    if atr_cfg:
        w = int(atr_cfg.get("window", 14))
        prev_close = df["close"].shift(1)
        tr = np.maximum(
            df["high"] - df["low"],
            np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()),
        )
        df[f"atr_{w}"] = tr.rolling(w).mean()

    # --- Volume features ---
    vol_cfg = cfg["features"].get("volume", {})
    if vol_cfg.get("log_volume", False):
        df["log_vol"] = np.log1p(df["volume"])
    for w in vol_cfg.get("vol_sma", []):
        df[f"vol_sma_{w}"] = df["volume"].rolling(w).mean() / (df["volume"] + 1e-9)

    # --- Calendar features ---
    cal = cfg["features"].get("calendar", {})
    if cal.get("day_of_week", False):
        df["dow"] = df["ts"].dt.dayofweek
        df = pd.get_dummies(df, columns=["dow"], drop_first=False)
        dow_cols = [c for c in df.columns if c.startswith("dow_")]
        df[dow_cols] = df[dow_cols].astype("int8")

    # --- Clean up ---
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df