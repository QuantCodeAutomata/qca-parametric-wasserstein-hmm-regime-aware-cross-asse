"""
Mean-Variance Optimization with turnover penalty.
Implements transaction-cost-aware portfolio optimization.
"""
import numpy as np
import cvxpy as cp
from typing import Optional

from config import ExperimentConfig


class MVOOptimizer:
    """
    Mean-variance optimizer with turnover penalty.
    
    Solves:
    max_w  mu^T w - gamma * w^T Sigma w - tau * ||w - w_prev||_1
    s.t.   1^T w = 1
           w >= 0
           w <= w_max
    """
    
    def __init__(
        self,
        gamma: float = 1.0,
        tau: float = 0.001,
        w_max: float = 0.6,
        solver: str = 'CLARABEL',
        solver_tol: float = 1e-8
    ):
        """
        Initialize MVO optimizer.
        
        Args:
            gamma: Risk aversion coefficient
            tau: Turnover penalty coefficient
            w_max: Maximum weight per asset
            solver: CVXPY solver name
            solver_tol: Solver tolerance
        """
        self.gamma = gamma
        self.tau = tau
        self.w_max = w_max
        self.solver = solver
        self.solver_tol = solver_tol
    
    def optimize(
        self,
        mu: np.ndarray,
        Sigma: np.ndarray,
        w_prev: np.ndarray
    ) -> np.ndarray:
        """
        Solve mean-variance optimization problem.
        
        Args:
            mu: Expected returns (n_assets,)
            Sigma: Covariance matrix (n_assets, n_assets)
            w_prev: Previous weights (n_assets,)
            
        Returns:
            Optimal weights (n_assets,)
        """
        n_assets = len(mu)
        
        # Decision variable
        w = cp.Variable(n_assets)
        
        # Auxiliary variables for L1 penalty: u_i >= |w_i - w_prev_i|
        u = cp.Variable(n_assets)
        
        # Objective: mu^T w - gamma * w^T Sigma w - tau * sum(u)
        portfolio_return = mu @ w
        portfolio_risk = cp.quad_form(w, Sigma)
        turnover_penalty = cp.sum(u)
        
        objective = cp.Maximize(portfolio_return - self.gamma * portfolio_risk - self.tau * turnover_penalty)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,           # Fully invested
            w >= 0,                   # Long-only
            w <= self.w_max,          # Position limits
            u >= w - w_prev,          # L1 penalty reformulation
            u >= w_prev - w,          # L1 penalty reformulation
            u >= 0                    # Auxiliary variable non-negativity
        ]
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            if self.solver == 'CLARABEL':
                problem.solve(solver=cp.CLARABEL, tol_gap_abs=self.solver_tol, tol_gap_rel=self.solver_tol)
            elif self.solver == 'OSQP':
                problem.solve(solver=cp.OSQP, eps_abs=self.solver_tol, eps_rel=self.solver_tol)
            else:
                problem.solve(solver=self.solver)
            
            if problem.status not in ['optimal', 'optimal_inaccurate']:
                # Fall back to equal-weight if optimization fails
                print(f"Warning: Optimization failed with status {problem.status}, using previous weights")
                return w_prev.copy()
            
            w_opt = w.value
            
            # Numerical cleanup
            w_opt = np.maximum(w_opt, 0)  # Ensure non-negative
            w_opt = np.minimum(w_opt, self.w_max)  # Ensure <= w_max
            w_opt = w_opt / w_opt.sum()  # Re-normalize
            
            return w_opt
            
        except Exception as e:
            print(f"Warning: Optimization error: {e}, using previous weights")
            return w_prev.copy()


def compute_turnover(w_curr: np.ndarray, w_prev: np.ndarray) -> float:
    """
    Compute turnover as 0.5 * ||w_curr - w_prev||_1.
    
    Args:
        w_curr: Current weights
        w_prev: Previous weights
        
    Returns:
        Turnover
    """
    return 0.5 * np.sum(np.abs(w_curr - w_prev))


def compute_effective_positions(w: np.ndarray) -> float:
    """
    Compute effective number of positions as 1 / sum(w_i^2).
    
    Args:
        w: Portfolio weights
        
    Returns:
        Effective number of positions
    """
    return 1.0 / np.sum(w ** 2)
