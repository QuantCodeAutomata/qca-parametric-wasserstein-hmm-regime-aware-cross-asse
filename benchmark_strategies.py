"""
Benchmark strategies for comparison.
Implements SPX buy-and-hold and equal-weight portfolio.
"""
import numpy as np
import pandas as pd
from typing import Dict

from config import ExperimentConfig


class BenchmarkStrategies:
    """
    Passive benchmark strategies for comparison.
    """
    
    def __init__(self, config: ExperimentConfig = ExperimentConfig):
        """
        Initialize benchmarks.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
    
    def run_spx_buyhold(
        self,
        returns: pd.DataFrame,
        test_start_idx: int
    ) -> pd.DataFrame:
        """
        Run SPX buy-and-hold strategy.
        
        100% allocation to SPX every day.
        
        Args:
            returns: Asset returns
            test_start_idx: First index of test period
            
        Returns:
            DataFrame with daily results
        """
        print("=" * 80)
        print("SPX BUY-AND-HOLD BENCHMARK")
        print("=" * 80)
        
        # Extract OOS period
        returns_oos = returns.iloc[test_start_idx:]
        
        # Portfolio return = SPX return
        r_port = returns_oos['SPX'].values
        
        # Weights: 100% SPX
        weights = {
            'w_SPX': np.ones(len(r_port)),
            'w_BOND': np.zeros(len(r_port)),
            'w_GOLD': np.zeros(len(r_port)),
            'w_OIL': np.zeros(len(r_port)),
            'w_USD': np.zeros(len(r_port))
        }
        
        # Turnover: 0 (static)
        turnover = np.zeros(len(r_port))
        
        # Effective positions: 1.0 (single asset)
        n_eff = np.ones(len(r_port))
        
        results_df = pd.DataFrame({
            'date': returns_oos.index,
            'r_port': r_port,
            'turnover': turnover,
            'n_eff': n_eff,
            **weights
        })
        results_df.set_index('date', inplace=True)
        
        print(f"Benchmark complete: {len(results_df)} days")
        print("=" * 80)
        
        return results_df
    
    def run_equalweight(
        self,
        returns: pd.DataFrame,
        test_start_idx: int
    ) -> pd.DataFrame:
        """
        Run equal-weight portfolio strategy.
        
        20% allocation to each of 5 assets every day.
        
        Args:
            returns: Asset returns
            test_start_idx: First index of test period
            
        Returns:
            DataFrame with daily results
        """
        print("=" * 80)
        print("EQUAL-WEIGHT CROSS-ASSET BENCHMARK")
        print("=" * 80)
        
        # Extract OOS period
        returns_oos = returns.iloc[test_start_idx:]
        
        # Portfolio return = mean of all asset returns
        asset_returns = returns_oos[self.config.ASSET_NAMES].values
        r_port = asset_returns.mean(axis=1)
        
        # Weights: 20% each
        weights = {
            'w_SPX': np.ones(len(r_port)) * 0.2,
            'w_BOND': np.ones(len(r_port)) * 0.2,
            'w_GOLD': np.ones(len(r_port)) * 0.2,
            'w_OIL': np.ones(len(r_port)) * 0.2,
            'w_USD': np.ones(len(r_port)) * 0.2
        }
        
        # Turnover: 0 (static, no rebalancing)
        turnover = np.zeros(len(r_port))
        
        # Effective positions: 5.0 (maximum diversification)
        n_eff = np.ones(len(r_port)) * 5.0
        
        results_df = pd.DataFrame({
            'date': returns_oos.index,
            'r_port': r_port,
            'turnover': turnover,
            'n_eff': n_eff,
            **weights
        })
        results_df.set_index('date', inplace=True)
        
        print(f"Benchmark complete: {len(results_df)} days")
        print("=" * 80)
        
        return results_df
