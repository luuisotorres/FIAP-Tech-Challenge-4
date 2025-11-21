from abc import ABC, abstractmethod
import pandas as pd

from fiap_tech_challenge_4.config import DataStrategyConfig
import fiap_tech_challenge_4.features.features as F


class DataStrategy(ABC):
    """
    Abstract base class for data processing strategies.

    Defines the interface for applying features to raw data.
    """

    def __init__(self, config: DataStrategyConfig):
        """
        Initializes the strategy with configuration.

        Args:
            config (DataStrategyConfig): The strategy configuration.
        """
        self.cfg = config

    @abstractmethod
    def apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering logic to the DataFrame.

        Args:
            df (pd.DataFrame): Raw input data.

        Returns:
            pd.DataFrame: Processed data with new features.
        """
        pass


class TrendFollowingStrategy(DataStrategy):
    """
    Strategy that emphasizes trend direction and momentum.

    Features:
    - Log Returns
    - Moving Average Ratios
    - MACD
    - Rolling Momentum
    """

    def apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Base
        df = F.add_log_return(df)

        # Technicals
        tech = self.cfg.technicals
        df = F.add_moving_averages(df, "close", sma_windows=tech.sma)

        if tech.macd:
            df = F.add_macd(
                df,
                fast=tech.macd.fast,
                slow=tech.macd.slow,
                signal=tech.macd.signal
            )

        # Momentum
        df = F.add_rolling_momentum(
            df,
            col="r_close",
            windows=self.cfg.rolling_windows
        )

        return F.clean_data(df)


class MeanReversionStrategy(DataStrategy):
    """
    Strategy that emphasizes overbought/oversold conditions and volatility.

    Features:
    - Log Returns
    - RSI
    - Rolling Volatility
    """

    def apply_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Base
        df = F.add_log_return(df)

        # RSI (Fixed window of 14 is standard, but could be parameterized)
        df = F.add_rsi(df, window=14)

        # Volatility
        df = F.add_rolling_volatility(
            df,
            col="r_close",
            windows=self.cfg.rolling_windows
        )

        return F.clean_data(df)
