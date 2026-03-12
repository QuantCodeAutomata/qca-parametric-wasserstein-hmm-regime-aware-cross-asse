"""
Data loading and feature construction module.
Implements strict-causal feature engineering for all experiments.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig


class DataLoader:
    """
    Load and prepare data for cross-asset allocation experiments.
    Implements strict causality: features for decision at day t use only data through t-1.
    """
    
    def __init__(self, config: ExperimentConfig = ExperimentConfig):
        """
        Initialize data loader.
        
        Args:
            config: Experiment configuration object
        """
        self.config = config
        self.prices_df = None
        self.returns_df = None
        self.features_df = None
        self.features_decision_df = None
        
    def download_data(self) -> pd.DataFrame:
        """
        Download adjusted close prices for all assets from Yahoo Finance.
        
        Returns:
            DataFrame with adjusted close prices, indexed by date
        """
        print("Downloading data from Yahoo Finance...")
        
        prices_dict = {}
        for asset_name, ticker in self.config.TICKERS.items():
            print(f"  Downloading {asset_name} ({ticker})...")
            try:
                data = yf.download(
                    ticker,
                    start=self.config.DATA_START_DATE,
                    end=self.config.DATA_END_DATE,
                    progress=False,
                    auto_adjust=True
                )
                
                # Handle both single and multi-index columns
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Adj Close' in data.columns.get_level_values(0):
                        prices_dict[asset_name] = data['Adj Close'].iloc[:, 0]
                    elif 'Close' in data.columns.get_level_values(0):
                        prices_dict[asset_name] = data['Close'].iloc[:, 0]
                else:
                    if 'Adj Close' in data.columns:
                        prices_dict[asset_name] = data['Adj Close']
                    elif 'Close' in data.columns:
                        prices_dict[asset_name] = data['Close']
                    else:
                        # Try first column if no Close column
                        prices_dict[asset_name] = data.iloc[:, 0]
            except Exception as e:
                print(f"    Warning: Failed to download {ticker}: {e}")
                prices_dict[asset_name] = None
        
        # Remove None values
        prices_dict = {k: v for k, v in prices_dict.items() if v is not None}
        
        # Create DataFrame and align on intersection calendar
        prices_df = pd.DataFrame(prices_dict)
        
        # Drop rows with any missing values (intersection method)
        initial_len = len(prices_df)
        prices_df = prices_df.dropna()
        final_len = len(prices_df)
        
        print(f"Data alignment: {initial_len} -> {final_len} days (dropped {initial_len - final_len} days)")
        print(f"Date range: {prices_df.index[0]} to {prices_df.index[-1]}")
        
        self.prices_df = prices_df
        return prices_df
    
    def compute_returns(self) -> pd.DataFrame:
        """
        Compute daily log returns.
        
        Returns:
            DataFrame with log returns r_t = log(P_t) - log(P_{t-1})
        """
        if self.prices_df is None:
            raise ValueError("Must download data first")
        
        # Log returns
        returns_df = np.log(self.prices_df).diff()
        
        # Drop first row (NaN)
        returns_df = returns_df.iloc[1:]
        
        print(f"Computed returns: {len(returns_df)} days")
        
        self.returns_df = returns_df
        return returns_df
    
    def compute_features(self) -> pd.DataFrame:
        """
        Compute rolling features for each asset:
        - 60-day rolling standard deviation
        - 20-day rolling mean
        
        Returns:
            DataFrame with features [r_t, sigma_t, m_t] for each asset
        """
        if self.returns_df is None:
            raise ValueError("Must compute returns first")
        
        features_list = []
        
        for asset in self.config.ASSET_NAMES:
            # Current return
            r_t = self.returns_df[asset]
            
            # Rolling standard deviation (60-day)
            sigma_t = self.returns_df[asset].rolling(
                window=self.config.ROLLING_STD_WINDOW, min_periods=self.config.ROLLING_STD_WINDOW
            ).std()
            
            # Rolling mean (20-day)
            m_t = self.returns_df[asset].rolling(
                window=self.config.ROLLING_MEAN_WINDOW, min_periods=self.config.ROLLING_MEAN_WINDOW
            ).mean()
            
            # Store features
            features_list.append(r_t.rename(f'{asset}_r'))
            features_list.append(sigma_t.rename(f'{asset}_sigma'))
            features_list.append(m_t.rename(f'{asset}_m'))
        
        features_df = pd.concat(features_list, axis=1)
        
        # Drop rows with NaN (warm-up period)
        features_df = features_df.dropna()
        
        print(f"Computed features: {len(features_df)} days (after warm-up)")
        
        self.features_df = features_df
        return features_df
    
    def create_decision_features(self) -> pd.DataFrame:
        """
        Create strict-causal decision features.
        
        For decision at day t, use features from t-1:
        x_t^decision = [r_{t-1}, sigma_{t-1}, m_{t-1}]
        
        Returns:
            DataFrame with decision features (shifted by 1 day)
        """
        if self.features_df is None:
            raise ValueError("Must compute features first")
        
        # Shift features by 1 day to enforce strict causality
        features_decision_df = self.features_df.shift(1).dropna()
        
        print(f"Created decision features: {len(features_decision_df)} days")
        
        self.features_decision_df = features_decision_df
        return features_decision_df
    
    def get_train_test_split(self) -> Tuple[pd.Timestamp, int, int]:
        """
        Get train/test split information.
        
        Returns:
            Tuple of (split_date, train_end_idx, test_start_idx)
        """
        if self.features_decision_df is None:
            raise ValueError("Must create decision features first")
        
        split_date = pd.Timestamp(self.config.OOS_START_DATE)
        
        # Find closest date in index
        if split_date not in self.features_decision_df.index:
            # Find nearest date on or after split_date
            valid_dates = self.features_decision_df.index[self.features_decision_df.index >= split_date]
            if len(valid_dates) == 0:
                raise ValueError(f"No data available on or after {split_date}")
            split_date = valid_dates[0]
            print(f"Adjusted split date to nearest available: {split_date}")
        
        train_end_idx = self.features_decision_df.index.get_loc(split_date) - 1
        test_start_idx = train_end_idx + 1
        
        print(f"Train/test split: {split_date}")
        print(f"  Train: {self.features_decision_df.index[0]} to {self.features_decision_df.index[train_end_idx]} ({train_end_idx + 1} days)")
        print(f"  Test:  {self.features_decision_df.index[test_start_idx]} to {self.features_decision_df.index[-1]} ({len(self.features_decision_df) - test_start_idx} days)")
        
        return split_date, train_end_idx, test_start_idx
    
    def prepare_all(self) -> Dict[str, pd.DataFrame]:
        """
        Run full data preparation pipeline.
        
        Returns:
            Dictionary with all prepared data
        """
        print("=" * 80)
        print("DATA PREPARATION PIPELINE")
        print("=" * 80)
        
        # Download and process
        self.download_data()
        self.compute_returns()
        self.compute_features()
        self.create_decision_features()
        
        # Get split info
        split_date, train_end_idx, test_start_idx = self.get_train_test_split()
        
        print("=" * 80)
        print(f"Data preparation complete!")
        print("=" * 80)
        
        return {
            'prices': self.prices_df,
            'returns': self.returns_df,
            'features': self.features_df,
            'features_decision': self.features_decision_df,
            'split_date': split_date,
            'train_end_idx': train_end_idx,
            'test_start_idx': test_start_idx,
        }


def load_data() -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load and prepare all data.
    
    Returns:
        Dictionary with all prepared data
    """
    loader = DataLoader()
    return loader.prepare_all()
