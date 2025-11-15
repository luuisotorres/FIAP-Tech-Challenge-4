import yfinance as yf
import pandas as pd


def fetch_data(ticker, period, interval, adjusted, multi_level_index):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=adjusted,
        multi_level_index=multi_level_index,
    )
    if df.empty:
        raise ValueError(
            f"No data found for ticker {ticker} with period {period} and interval {interval}."
        )
    df["ts"] = pd.to_datetime(df.index)
    df.columns = df.columns.str.lower()
    df = df.reset_index(drop=True)[["ts", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("ts").drop_duplicates("ts")
    df.dropna(inplace=True)
    return df
