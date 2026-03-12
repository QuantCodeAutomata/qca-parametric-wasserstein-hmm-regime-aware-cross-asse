"""
Parametric Wasserstein HMM strategy implementation.
Implements template tracking with Wasserstein geometry for regime-aware allocation.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig
from hmm_utils import GaussianHMMWrapper, model_order_selection
from wasserstein_utils import map_components_to_templates, make_positive_definite
from mvo_optimizer import MVOOptimizer, compute_turnover, compute_effective_positions


class ParametricWassersteinHMMStrategy:
    """
    Parametric Wasserstein HMM strategy for cross-asset allocation.
    
    Implements:
    - Predictive model-order selection
    - Gaussian HMM inference
    - Wasserstein-distance-based template tracking
    - Template-conditional moment aggregation
    - Transaction-cost-aware MVO optimization
    """
    
    def __init__(self, config: ExperimentConfig = ExperimentConfig):
        """
        Initialize strategy.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.optimizer = MVOOptimizer(
            gamma=config.MVO.GAMMA,
            tau=config.MVO.TAU,
            w_max=config.MVO.W_MAX,
            solver=config.MVO.SOLVER,
            solver_tol=config.MVO.SOLVER_TOL
        )
        
        # Templates {(mu_g, Sigma_g)}_{g=1}^G
        self.templates = None
        
        # Results storage
        self.results = []
        
    def initialize_templates(self, X_calib: np.ndarray) -> None:
        """
        Initialize templates from calibration window using G-state HMM.
        
        Args:
            X_calib: Calibration feature vectors
        """
        print(f"Initializing {self.config.ParametricHMM.N_TEMPLATES} templates from calibration window...")
        
        # Fit G-state HMM on calibration data
        model = GaussianHMMWrapper(
            n_components=self.config.ParametricHMM.N_TEMPLATES,
            covariance_type=self.config.ParametricHMM.HMM_COVARIANCE_TYPE,
            n_iter=self.config.ParametricHMM.HMM_MAX_ITER,
            tol=self.config.ParametricHMM.HMM_TOL,
            n_init=self.config.ParametricHMM.HMM_N_INIT,
            random_state=self.config.RANDOM_SEED
        )
        model.fit(X_calib)
        
        # Extract components as initial templates
        self.templates = model.get_components()
        
        print(f"  Initialized {len(self.templates)} templates")
        
    def run_backtest(
        self,
        features_decision: pd.DataFrame,
        returns: pd.DataFrame,
        train_end_idx: int,
        test_start_idx: int
    ) -> pd.DataFrame:
        """
        Run full backtest of parametric strategy.
        
        Args:
            features_decision: Strict-causal decision features
            returns: Asset returns (aligned with features)
            train_end_idx: Last index of training period
            test_start_idx: First index of test period
            
        Returns:
            DataFrame with daily results
        """
        print("=" * 80)
        print("PARAMETRIC WASSERSTEIN HMM STRATEGY BACKTEST")
        print("=" * 80)
        
        # Extract numpy arrays
        X_all = features_decision.values
        R_all = returns.loc[features_decision.index, self.config.ASSET_NAMES].values
        dates = features_decision.index
        
        # Initialize templates from calibration window
        X_calib = X_all[:train_end_idx+1]
        self.initialize_templates(X_calib)
        
        # Initialize weights
        w_prev = np.array(self.config.INITIAL_WEIGHTS)
        
        # Storage
        results = []
        
        # Track last model selection
        last_K = self.config.ParametricHMM.N_TEMPLATES
        last_model_selection_idx = train_end_idx
        
        # OOS loop
        for t_idx in range(test_start_idx, len(X_all)):
            date = dates[t_idx]
            
            # Expanding history (through t-1)
            X_history = X_all[:t_idx]
            
            # Model order selection (every F_K days)
            days_since_selection = t_idx - last_model_selection_idx
            if days_since_selection >= self.config.ParametricHMM.MODEL_SELECTION_FREQ:
                K_t, scores = model_order_selection(
                    X_history,
                    self.config.ParametricHMM.K_CANDIDATES,
                    self.config.ParametricHMM.VALIDATION_SIZE,
                    self.config.ParametricHMM.LAMBDA_K,
                    self.config
                )
                last_K = K_t
                last_model_selection_idx = t_idx
            else:
                K_t = last_K
            
            # Fit K_t-state HMM on history
            model = GaussianHMMWrapper(
                n_components=K_t,
                covariance_type=self.config.ParametricHMM.HMM_COVARIANCE_TYPE,
                n_iter=self.config.ParametricHMM.HMM_MAX_ITER,
                tol=self.config.ParametricHMM.HMM_TOL,
                n_init=self.config.ParametricHMM.HMM_N_INIT,
                random_state=self.config.RANDOM_SEED + t_idx
            )
            
            try:
                model.fit(X_history)
            except Exception as e:
                print(f"  Warning: HMM fitting failed at {date}, using previous weights")
                # Use previous weights
                w_t = w_prev.copy()
                r_t = R_all[t_idx]
                r_port = np.dot(w_t, r_t)
                turnover = compute_turnover(w_t, w_prev)
                n_eff = compute_effective_positions(w_t)
                
                results.append({
                    'date': date,
                    'K_t': K_t,
                    'r_port': r_port,
                    'turnover': turnover,
                    'n_eff': n_eff,
                    **{f'w_{asset}': w_t[i] for i, asset in enumerate(self.config.ASSET_NAMES)},
                    **{f'p_g{g}': 0.0 for g in range(self.config.ParametricHMM.N_TEMPLATES)},
                    'dominant_regime': -1
                })
                w_prev = w_t
                continue
            
            # Extract components
            components = model.get_components()
            
            # Compute predictive state probabilities
            pred_probs = model.compute_predictive_probabilities(X_history)
            p_t_k = pred_probs[-1]  # Last time step (predictive for t)
            
            # Map components to templates via Wasserstein distance
            mapping, distances = map_components_to_templates(components, self.templates)
            
            # Aggregate template probabilities
            p_t_g = np.zeros(self.config.ParametricHMM.N_TEMPLATES)
            for k in range(K_t):
                g = mapping[k]
                p_t_g[g] += p_t_k[k]
            
            # Ensure normalization
            p_t_g = p_t_g / p_t_g.sum()
            
            # Update templates with exponential smoothing
            eta = self.config.ParametricHMM.ETA
            for g in range(self.config.ParametricHMM.N_TEMPLATES):
                if p_t_g[g] > 0:
                    # Find components assigned to template g
                    assigned_components = [k for k in range(K_t) if mapping[k] == g]
                    
                    if len(assigned_components) > 0:
                        # Compute normalized weights
                        assigned_probs = np.array([p_t_k[k] for k in assigned_components])
                        alpha_k = assigned_probs / assigned_probs.sum()
                        
                        # Aggregate moments
                        mu_bar = sum(alpha_k[i] * components[k][0] for i, k in enumerate(assigned_components))
                        Sigma_bar = sum(alpha_k[i] * components[k][1] for i, k in enumerate(assigned_components))
                        
                        # Update template
                        mu_g_old, Sigma_g_old = self.templates[g]
                        mu_g_new = (1 - eta) * mu_g_old + eta * mu_bar
                        Sigma_g_new = (1 - eta) * Sigma_g_old + eta * Sigma_bar
                        
                        # Regularize
                        Sigma_g_new = make_positive_definite(Sigma_g_new, self.config.ParametricHMM.COV_JITTER)
                        
                        self.templates[g] = (mu_g_new, Sigma_g_new)
            
            # Aggregate template moments
            mu_t = sum(p_t_g[g] * self.templates[g][0] for g in range(self.config.ParametricHMM.N_TEMPLATES))
            Sigma_t = sum(p_t_g[g] * self.templates[g][1] for g in range(self.config.ParametricHMM.N_TEMPLATES))
            
            # Extract asset moments (first N=5 dimensions)
            mu_assets = mu_t[:self.config.N_ASSETS]
            Sigma_assets = Sigma_t[:self.config.N_ASSETS, :self.config.N_ASSETS]
            
            # Regularize
            Sigma_assets = make_positive_definite(Sigma_assets, self.config.ParametricHMM.COV_JITTER)
            
            # MVO optimization
            w_t = self.optimizer.optimize(mu_assets, Sigma_assets, w_prev)
            
            # Realize return
            r_t = R_all[t_idx]
            r_port = np.dot(w_t, r_t)
            
            # Compute diagnostics
            turnover = compute_turnover(w_t, w_prev)
            n_eff = compute_effective_positions(w_t)
            dominant_regime = np.argmax(p_t_g)
            
            # Store results
            results.append({
                'date': date,
                'K_t': K_t,
                'r_port': r_port,
                'turnover': turnover,
                'n_eff': n_eff,
                **{f'w_{asset}': w_t[i] for i, asset in enumerate(self.config.ASSET_NAMES)},
                **{f'p_g{g}': p_t_g[g] for g in range(self.config.ParametricHMM.N_TEMPLATES)},
                'dominant_regime': dominant_regime
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
