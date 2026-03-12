"""
KNN conditional-moment baseline strategy implementation.
Implements non-parametric regime inference via K-nearest neighbors.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import LedoitWolf
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig
from mvo_optimizer import MVOOptimizer, compute_turnover, compute_effective_positions
from wasserstein_utils import make_positive_definite


class KNNStrategy:
    """
    KNN conditional-moment baseline strategy.
    
    Implements:
    - K-nearest neighbor retrieval
    - Neighbor-averaged expected returns
    - Ledoit-Wolf shrinkage covariance estimation
    - Transaction-cost-aware MVO optimization (identical to parametric)
    """
    
    def __init__(
        self,
        k_neighbors: int = 50,
        config: ExperimentConfig = ExperimentConfig
    ):
        """
        Initialize KNN strategy.
        
        Args:
            k_neighbors: Number of nearest neighbors
            config: Experiment configuration
        """
        self.config = config
        self.k_neighbors = k_neighbors
        
        # Identical MVO optimizer as parametric strategy
        self.optimizer = MVOOptimizer(
            gamma=config.MVO.GAMMA,
            tau=config.MVO.TAU,
            w_max=config.MVO.W_MAX,
            solver=config.MVO.SOLVER,
            solver_tol=config.MVO.SOLVER_TOL
        )
    
    def run_backtest(
        self,
        features_decision: pd.DataFrame,
        returns: pd.DataFrame,
        train_end_idx: int,
        test_start_idx: int
    ) -> pd.DataFrame:
        """
        Run full backtest of KNN strategy.
        
        Args:
            features_decision: Strict-causal decision features
            returns: Asset returns (aligned with features)
            train_end_idx: Last index of training period
            test_start_idx: First index of test period
            
        Returns:
            DataFrame with daily results
        """
        print("=" * 80)
        print("KNN CONDITIONAL-MOMENT BASELINE STRATEGY BACKTEST")
        print("=" * 80)
        
        # Extract numpy arrays
        X_all = features_decision.values
        R_all = returns.loc[features_decision.index, self.config.ASSET_NAMES].values
        dates = features_decision.index
        
        # Initialize weights
        w_prev = np.array(self.config.INITIAL_WEIGHTS)
        
        # Storage
        results = []
        
        # OOS loop
        for t_idx in range(test_start_idx, len(X_all)):
            date = dates[t_idx]
            
            # Expanding history (through t-1)
            X_history = X_all[:t_idx]
            R_history = R_all[:t_idx]
            
            # Ensure sufficient neighbors
            if len(X_history) < self.k_neighbors:
                # Use previous weights
                w_t = w_prev.copy()
                r_t = R_all[t_idx]
                r_port = np.dot(w_t, r_t)
                turnover = compute_turnover(w_t, w_prev)
                n_eff = compute_effective_positions(w_t)
                
                results.append({
                    'date': date,
                    'r_port': r_port,
                    'turnover': turnover,
                    'n_eff': n_eff,
                    **{f'w_{asset}': w_t[i] for i, asset in enumerate(self.config.ASSET_NAMES)}
                })
                w_prev = w_t
                continue
            
            # Query vector (current feature for decision)
            q_t = X_all[t_idx].reshape(1, -1)
            
            # Find K nearest neighbors
            knn = NearestNeighbors(
                n_neighbors=self.k_neighbors,
                metric='euclidean',
                algorithm='ball_tree'
            )
            knn.fit(X_history)
            distances, indices = knn.kneighbors(q_t)
            
            # Extract neighbor returns
            neighbor_returns = R_history[indices[0]]  # Shape: (k_neighbors, n_assets)
            
            # Estimate moments
            # Expected returns: average of neighbor returns
            mu_t = neighbor_returns.mean(axis=0)
            
            # Covariance: Ledoit-Wolf shrinkage
            if len(neighbor_returns) >= self.config.N_ASSETS:
                try:
                    lw = LedoitWolf()
                    lw.fit(neighbor_returns)
                    Sigma_t = lw.covariance_
                except Exception as e:
                    # Fall back to sample covariance with jitter
                    Sigma_t = np.cov(neighbor_returns.T)
                    Sigma_t = make_positive_definite(Sigma_t, jitter=1e-6)
            else:
                # Not enough samples; use sample covariance with jitter
                Sigma_t = np.cov(neighbor_returns.T)
                Sigma_t = make_positive_definite(Sigma_t, jitter=1e-6)
            
            # MVO optimization
            w_t = self.optimizer.optimize(mu_t, Sigma_t, w_prev)
            
            # Realize return
            r_t = R_all[t_idx]
            r_port = np.dot(w_t, r_t)
            
            # Compute diagnostics
            turnover = compute_turnover(w_t, w_prev)
            n_eff = compute_effective_positions(w_t)
            
            # Store results
            results.append({
                'date': date,
                'r_port': r_port,
                'turnover': turnover,
                'n_eff': n_eff,
                **{f'w_{asset}': w_t[i] for i, asset in enumerate(self.config.ASSET_NAMES)}
            })
            
            # Update previous weights
            w_prev = w_t
            
            # Progress
            if (t_idx - test_start_idx + 1) % 50 == 0:
                print(f"  Processed {t_idx - test_start_idx + 1} / {len(X_all) - test_start_idx} days")
        
        results_df = pd.DataFrame(results)
        results_df.set_index('date', inplace=True)
        
        print(f"Backtest complete: {len(results_df)} days")
        print("=" * 80)
        
        return results_df
