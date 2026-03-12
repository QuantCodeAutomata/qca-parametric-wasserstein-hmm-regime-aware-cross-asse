"""
Configuration file for all experiments.
Records all parameters used in the Parametric Wasserstein HMM regime-aware cross-asset allocation experiments.
"""
import os
from datetime import datetime
from typing import Dict, Any


class ExperimentConfig:
    """Central configuration for all experiments."""
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Asset tickers (Yahoo Finance)
    TICKERS = {
        'SPX': '^GSPC',      # S&P 500 Index
        'BOND': 'TLT',       # iShares 20+ Year Treasury Bond ETF
        'GOLD': 'GLD',       # SPDR Gold Trust
        'OIL': 'USO',        # United States Oil Fund
        'USD': 'UUP'         # Invesco DB US Dollar Index Bullish Fund
    }
    
    ASSET_NAMES = ['SPX', 'BOND', 'GOLD', 'OIL', 'USD']
    N_ASSETS = 5
    
    # Date ranges
    DATA_START_DATE = '2005-01-01'
    DATA_END_DATE = '2026-02-28'
    OOS_START_DATE = '2023-05-01'  # Train/test split date t_0
    
    # Calendar alignment method
    CALENDAR_METHOD = 'intersection'  # Keep only dates where all assets have prices
    
    # Feature construction parameters
    ROLLING_STD_WINDOW = 60   # 60-day rolling standard deviation
    ROLLING_MEAN_WINDOW = 20  # 20-day rolling mean
    FEATURE_DIM = 15          # 5 assets * 3 features (return, std, mean)
    
    # Minimum warm-up periods
    MIN_WARMUP_DAYS = 60           # For feature construction
    MIN_CALIBRATION_DAYS = 252     # For initial template calibration (1 year)
    
    # Parametric HMM Strategy (exp_1) parameters
    class ParametricHMM:
        """Parameters for Parametric Wasserstein HMM strategy."""
        
        # Template configuration
        N_TEMPLATES = 6  # G = 6 templates
        
        # Model order selection
        MODEL_SELECTION_FREQ = 5  # F_K = 5 days between model selection
        VALIDATION_SIZE = 60      # |V| = 60 validation points
        K_CANDIDATES = [2, 3, 4, 5, 6, 7, 8]  # Candidate HMM states
        LAMBDA_K = 1.0  # Complexity penalty coefficient
        
        # HMM fitting parameters
        HMM_COVARIANCE_TYPE = 'full'
        HMM_MAX_ITER = 200
        HMM_TOL = 1e-4
        HMM_N_INIT = 5  # Number of random restarts
        
        # Template update parameters
        ETA = 0.05  # Exponential smoothing rate for template updates
        
        # Covariance regularization
        COV_JITTER = 1e-6  # Diagonal jitter for non-PD covariance matrices
        
    # KNN Strategy (exp_2) parameters
    class KNN:
        """Parameters for KNN conditional-moment baseline."""
        
        K_NEIGHBORS = 50  # Number of nearest neighbors
        DISTANCE_METRIC = 'euclidean'
        FEATURE_SCALING = 'none'  # No scaling by default
        
        # Sensitivity analysis values
        K_SENSITIVITY = [20, 50, 100, 200]
    
    # Mean-Variance Optimization (MVO) parameters (shared across strategies)
    class MVO:
        """Mean-variance optimization parameters."""
        
        GAMMA = 1.0        # Risk aversion coefficient
        TAU = 0.001        # Turnover penalty coefficient
        W_MAX = 0.6        # Maximum weight per asset
        
        # Solver configuration
        SOLVER = 'CLARABEL'  # CVXPY solver
        SOLVER_TOL = 1e-8
        
    # Initial weights (equal-weight)
    INITIAL_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    # Ablation study configurations
    class Ablations:
        """Ablation study parameter variations."""
        
        ETA_VALUES = [0.01, 0.05, 0.1, 0.2]
        FIXED_K = 4  # For fixed-K ablation
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging."""
        config = {
            'random_seed': cls.RANDOM_SEED,
            'tickers': cls.TICKERS,
            'asset_names': cls.ASSET_NAMES,
            'data_start': cls.DATA_START_DATE,
            'data_end': cls.DATA_END_DATE,
            'oos_start': cls.OOS_START_DATE,
            'calendar_method': cls.CALENDAR_METHOD,
            'rolling_std_window': cls.ROLLING_STD_WINDOW,
            'rolling_mean_window': cls.ROLLING_MEAN_WINDOW,
            'feature_dim': cls.FEATURE_DIM,
            'parametric_hmm': {
                'n_templates': cls.ParametricHMM.N_TEMPLATES,
                'model_selection_freq': cls.ParametricHMM.MODEL_SELECTION_FREQ,
                'validation_size': cls.ParametricHMM.VALIDATION_SIZE,
                'k_candidates': cls.ParametricHMM.K_CANDIDATES,
                'lambda_k': cls.ParametricHMM.LAMBDA_K,
                'hmm_covariance_type': cls.ParametricHMM.HMM_COVARIANCE_TYPE,
                'hmm_max_iter': cls.ParametricHMM.HMM_MAX_ITER,
                'hmm_tol': cls.ParametricHMM.HMM_TOL,
                'hmm_n_init': cls.ParametricHMM.HMM_N_INIT,
                'eta': cls.ParametricHMM.ETA,
                'cov_jitter': cls.ParametricHMM.COV_JITTER,
            },
            'knn': {
                'k_neighbors': cls.KNN.K_NEIGHBORS,
                'distance_metric': cls.KNN.DISTANCE_METRIC,
                'feature_scaling': cls.KNN.FEATURE_SCALING,
            },
            'mvo': {
                'gamma': cls.MVO.GAMMA,
                'tau': cls.MVO.TAU,
                'w_max': cls.MVO.W_MAX,
                'solver': cls.MVO.SOLVER,
                'solver_tol': cls.MVO.SOLVER_TOL,
            },
            'initial_weights': cls.INITIAL_WEIGHTS,
        }
        return config
    
    @classmethod
    def save_config(cls, filepath: str = 'results/experiment_config.json') -> None:
        """Save configuration to JSON file."""
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(cls.to_dict(), f, indent=2)
        print(f"Configuration saved to {filepath}")
