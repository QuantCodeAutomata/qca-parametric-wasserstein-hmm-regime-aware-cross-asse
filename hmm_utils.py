"""
HMM utilities for regime inference.
Implements predictive model-order selection and forward algorithm.
"""
import numpy as np
from hmmlearn import hmm
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig
from wasserstein_utils import make_positive_definite


class GaussianHMMWrapper:
    """
    Wrapper for hmmlearn GaussianHMM with additional utilities.
    """
    
    def __init__(
        self,
        n_components: int,
        covariance_type: str = 'full',
        n_iter: int = 200,
        tol: float = 1e-4,
        n_init: int = 5,
        random_state: int = 42
    ):
        """
        Initialize Gaussian HMM.
        
        Args:
            n_components: Number of hidden states
            covariance_type: Type of covariance ('full', 'diag', 'spherical', 'tied')
            n_iter: Maximum number of EM iterations
            tol: Convergence tolerance
            n_init: Number of random initializations
            random_state: Random seed
        """
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            init_params='stmc',
            params='stmc'
        )
        self.n_init = n_init
        self.random_state = random_state
        self.best_loglik = -np.inf
        self.fitted = False
    
    def fit(self, X: np.ndarray) -> 'GaussianHMMWrapper':
        """
        Fit HMM with multiple random initializations.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        best_model = None
        best_loglik = -np.inf
        
        for init_idx in range(self.n_init):
            # Set random seed for this initialization
            seed = self.random_state + init_idx
            
            # Create new model
            model = hmm.GaussianHMM(
                n_components=self.n_components,
                covariance_type=self.model.covariance_type,
                n_iter=self.model.n_iter,
                tol=self.model.tol,
                random_state=seed,
                init_params='stmc',
                params='stmc'
            )
            
            try:
                model.fit(X)
                loglik = model.score(X)
                
                if loglik > best_loglik:
                    best_loglik = loglik
                    best_model = model
            except Exception as e:
                # Skip failed initializations
                continue
        
        if best_model is None:
            raise ValueError(f"All {self.n_init} initializations failed for K={self.n_components}")
        
        self.model = best_model
        self.best_loglik = best_loglik
        self.fitted = True
        
        return self
    
    def get_components(self) -> list:
        """
        Extract fitted Gaussian components.
        
        Returns:
            List of (mu_k, Sigma_k) tuples for each component
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        components = []
        for k in range(self.n_components):
            mu_k = self.model.means_[k]
            Sigma_k = self.model.covars_[k]
            
            # Regularize covariance
            Sigma_k = make_positive_definite(Sigma_k, jitter=ExperimentConfig.ParametricHMM.COV_JITTER)
            
            components.append((mu_k, Sigma_k))
        
        return components
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get transition matrix."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.model.transmat_
    
    def get_initial_distribution(self) -> np.ndarray:
        """Get initial state distribution."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.model.startprob_
    
    def compute_forward_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Compute forward probabilities alpha_t,k = P(x_1,...,x_t, z_t=k | params).
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Forward probabilities of shape (n_samples, n_components)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Manual implementation of forward algorithm
        n_samples = X.shape[0]
        n_components = self.n_components
        
        # Compute emission probabilities for all time steps
        from scipy.stats import multivariate_normal
        emission_probs = np.zeros((n_samples, n_components))
        for k in range(n_components):
            # Regularize covariance for numerical stability
            cov_k = make_positive_definite(self.model.covars_[k], jitter=1e-6)
            try:
                emission_probs[:, k] = multivariate_normal.pdf(
                    X, mean=self.model.means_[k], cov=cov_k, allow_singular=True
                )
            except:
                # Fall back to uniform if computation fails
                emission_probs[:, k] = 1.0 / n_components
        
        # Forward algorithm
        alpha = np.zeros((n_samples, n_components))
        
        # Initialize
        alpha[0] = self.model.startprob_ * emission_probs[0]
        alpha[0] = alpha[0] / alpha[0].sum()
        
        # Forward pass
        for t in range(1, n_samples):
            alpha[t] = (alpha[t-1] @ self.model.transmat_) * emission_probs[t]
            alpha[t] = alpha[t] / alpha[t].sum()
        
        return alpha
    
    def compute_predictive_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictive state probabilities p_{t,k} = P(z_t=k | x_1,...,x_{t-1}).
        
        For each t, compute: p_{t,k} = sum_j alpha_{t-1,j} * A[j,k]
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Predictive probabilities of shape (n_samples, n_components)
            First row is the initial distribution (no conditioning)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Get forward probabilities
        alpha = self.compute_forward_probabilities(X)
        
        # Get transition matrix
        A = self.get_transition_matrix()
        
        # Compute predictive probabilities
        n_samples = X.shape[0]
        pred_probs = np.zeros((n_samples, self.n_components))
        
        # First time step: use initial distribution
        pred_probs[0] = self.get_initial_distribution()
        
        # Subsequent time steps: p_{t,k} = sum_j alpha_{t-1,j} * A[j,k]
        for t in range(1, n_samples):
            pred_probs[t] = alpha[t-1] @ A
        
        # Normalize (should already be normalized, but ensure numerical stability)
        pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)
        
        return pred_probs
    
    def compute_validation_loglik(self, X_train: np.ndarray, X_val: np.ndarray) -> float:
        """
        Compute predictive log-likelihood on validation set.
        
        For each validation point s, compute log p(x_s | x_1,...,x_{s-1}, params).
        
        Args:
            X_train: Training data (before validation set)
            X_val: Validation data
            
        Returns:
            Sum of predictive log-likelihoods over validation set
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Concatenate train and validation
        X_full = np.vstack([X_train, X_val])
        
        # Compute emission probabilities
        from scipy.stats import multivariate_normal
        n_samples = X_full.shape[0]
        n_components = self.n_components
        
        emission_probs = np.zeros((n_samples, n_components))
        for k in range(n_components):
            # Regularize covariance for numerical stability
            cov_k = make_positive_definite(self.model.covars_[k], jitter=1e-6)
            try:
                emission_probs[:, k] = multivariate_normal.pdf(
                    X_full, mean=self.model.means_[k], cov=cov_k, allow_singular=True
                )
            except:
                emission_probs[:, k] = 1.0 / n_components
        
        # Compute forward probabilities
        alpha = np.zeros((n_samples, n_components))
        alpha[0] = self.model.startprob_ * emission_probs[0]
        
        n_train = X_train.shape[0]
        val_loglik = 0.0
        
        for t in range(1, n_samples):
            alpha[t] = (alpha[t-1] @ self.model.transmat_) * emission_probs[t]
            
            # If in validation set, accumulate log-likelihood
            if t >= n_train:
                val_loglik += np.log(np.sum(alpha[t]) + 1e-300)
            
            # Normalize
            alpha[t] = alpha[t] / (np.sum(alpha[t]) + 1e-300)
        
        return val_loglik


def model_order_selection(
    X: np.ndarray,
    k_candidates: list,
    validation_size: int,
    lambda_k: float,
    config: ExperimentConfig = ExperimentConfig
) -> Tuple[int, Dict[int, float]]:
    """
    Perform model order selection with predictive validation.
    
    For each candidate K:
    1. Fit K-state HMM on full history X
    2. Compute predictive log-likelihood on last validation_size points
    3. Apply complexity penalty: score(K) = PredLL(K) - lambda_k * complexity(K)
    4. Select K with highest score
    
    Args:
        X: Feature history of shape (n_samples, n_features)
        k_candidates: List of candidate number of states
        validation_size: Number of validation points (from end of X)
        lambda_k: Complexity penalty coefficient
        config: Experiment configuration
        
    Returns:
        Tuple of (best_k, scores_dict)
    """
    if len(X) < validation_size + 50:
        # Not enough data; use median K
        return int(np.median(k_candidates)), {}
    
    # Split into train and validation
    X_train = X[:-validation_size]
    X_val = X[-validation_size:]
    
    scores = {}
    
    for K in k_candidates:
        try:
            # Fit HMM on full history
            model = GaussianHMMWrapper(
                n_components=K,
                covariance_type=config.ParametricHMM.HMM_COVARIANCE_TYPE,
                n_iter=config.ParametricHMM.HMM_MAX_ITER,
                tol=config.ParametricHMM.HMM_TOL,
                n_init=config.ParametricHMM.HMM_N_INIT,
                random_state=config.RANDOM_SEED
            )
            model.fit(X)
            
            # Compute validation log-likelihood
            val_loglik = model.compute_validation_loglik(X_train, X_val)
            
            # Complexity penalty: number of free parameters
            # K means (d-dimensional) + K covariances (d x d) + (K-1) transition parameters
            d = X.shape[1]
            n_mean_params = K * d
            n_cov_params = K * d * (d + 1) // 2  # Full covariance
            n_transition_params = K * (K - 1)  # Transition matrix
            complexity = n_mean_params + n_cov_params + n_transition_params
            
            # Score with penalty
            score = val_loglik - lambda_k * complexity
            scores[K] = score
            
        except Exception as e:
            # Skip failed fits
            scores[K] = -np.inf
    
    # Select best K
    if len(scores) == 0 or all(s == -np.inf for s in scores.values()):
        # All failed; use median
        best_k = int(np.median(k_candidates))
    else:
        best_k = max(scores, key=scores.get)
    
    return best_k, scores
