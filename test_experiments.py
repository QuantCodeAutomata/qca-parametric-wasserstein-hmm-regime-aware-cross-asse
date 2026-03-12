"""
Comprehensive tests for all experiment components.
Tests validate methodology adherence and correctness.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import linalg

from config import ExperimentConfig
from wasserstein_utils import (
    symmetrize_matrix,
    make_positive_definite,
    matrix_sqrt,
    wasserstein2_squared,
    wasserstein2
)
from mvo_optimizer import MVOOptimizer, compute_turnover, compute_effective_positions
from hmm_utils import GaussianHMMWrapper
from performance_metrics import PerformanceAnalyzer


class TestWassersteinUtils:
    """Test Wasserstein distance utilities."""
    
    def test_symmetrize_matrix(self):
        """Test matrix symmetrization."""
        A = np.array([[1, 2], [3, 4]])
        A_sym = symmetrize_matrix(A)
        
        # Should be symmetric
        assert np.allclose(A_sym, A_sym.T)
        
        # Check values
        expected = np.array([[1, 2.5], [2.5, 4]])
        assert np.allclose(A_sym, expected)
    
    def test_make_positive_definite(self):
        """Test positive definite enforcement."""
        # Start with non-PD matrix (negative eigenvalue)
        A = np.array([[1, 2], [2, 1]])
        # This matrix has eigenvalues: 3 and -1, so it's not PD
        
        A_pd = make_positive_definite(A, jitter=1e-3)
        
        # Should be positive definite (Cholesky succeeds)
        try:
            np.linalg.cholesky(A_pd)
            is_pd = True
        except np.linalg.LinAlgError:
            is_pd = False
        
        assert is_pd, "Matrix should be positive definite after regularization"
    
    def test_matrix_sqrt(self):
        """Test matrix square root computation."""
        # Create a positive definite matrix
        A = np.array([[4, 1], [1, 3]])
        
        A_sqrt = matrix_sqrt(A)
        
        # Verify A_sqrt @ A_sqrt ≈ A
        reconstructed = A_sqrt @ A_sqrt
        assert np.allclose(reconstructed, A, atol=1e-6)
        
        # Should be symmetric
        assert np.allclose(A_sqrt, A_sqrt.T, atol=1e-6)
    
    def test_wasserstein2_squared_identical(self):
        """Test Wasserstein distance between identical distributions."""
        mu = np.array([0, 0])
        Sigma = np.eye(2)
        
        dist_sq = wasserstein2_squared(mu, Sigma, mu, Sigma)
        
        # Distance should be zero
        assert dist_sq < 1e-10
    
    def test_wasserstein2_squared_mean_only(self):
        """Test Wasserstein distance with only mean difference."""
        mu1 = np.array([0, 0])
        mu2 = np.array([1, 0])
        Sigma = np.eye(2)
        
        dist_sq = wasserstein2_squared(mu1, Sigma, mu2, Sigma)
        
        # Distance squared should be ||mu1 - mu2||^2 = 1
        assert np.isclose(dist_sq, 1.0, atol=1e-6)
    
    def test_wasserstein2(self):
        """Test Wasserstein distance (non-squared)."""
        mu1 = np.array([0, 0])
        mu2 = np.array([3, 4])
        Sigma = np.eye(2)
        
        dist = wasserstein2(mu1, Sigma, mu2, Sigma)
        
        # Distance should be ||mu1 - mu2||_2 = 5
        assert np.isclose(dist, 5.0, atol=1e-6)


class TestMVOOptimizer:
    """Test mean-variance optimization."""
    
    def test_optimize_basic(self):
        """Test basic optimization."""
        optimizer = MVOOptimizer(gamma=1.0, tau=0.001, w_max=0.6)
        
        # Simple case: 3 assets
        mu = np.array([0.01, 0.005, 0.002])
        Sigma = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.02, 0.003],
            [0.005, 0.003, 0.01]
        ])
        w_prev = np.array([1/3, 1/3, 1/3])
        
        w_opt = optimizer.optimize(mu, Sigma, w_prev)
        
        # Check constraints
        assert np.isclose(np.sum(w_opt), 1.0, atol=1e-4)
        assert np.all(w_opt >= -1e-6)
        assert np.all(w_opt <= 0.6 + 1e-6)
    
    def test_optimize_equal_returns(self):
        """Test optimization with equal expected returns."""
        optimizer = MVOOptimizer(gamma=1.0, tau=0.001, w_max=0.6)
        
        # All assets have same expected return
        mu = np.array([0.01, 0.01, 0.01])
        Sigma = np.eye(3) * 0.01
        w_prev = np.array([1/3, 1/3, 1/3])
        
        w_opt = optimizer.optimize(mu, Sigma, w_prev)
        
        # Should be close to equal weight (due to turnover penalty)
        assert np.isclose(np.sum(w_opt), 1.0, atol=1e-4)
        assert np.all(w_opt >= -1e-6)
    
    def test_compute_turnover(self):
        """Test turnover computation."""
        w_prev = np.array([0.5, 0.3, 0.2])
        w_curr = np.array([0.4, 0.4, 0.2])
        
        turnover = compute_turnover(w_curr, w_prev)
        
        # Turnover = 0.5 * (|0.4-0.5| + |0.4-0.3| + |0.2-0.2|) = 0.5 * 0.2 = 0.1
        assert np.isclose(turnover, 0.1, atol=1e-6)
    
    def test_compute_turnover_no_change(self):
        """Test turnover with no change."""
        w = np.array([0.5, 0.3, 0.2])
        
        turnover = compute_turnover(w, w)
        
        assert np.isclose(turnover, 0.0, atol=1e-10)
    
    def test_compute_effective_positions(self):
        """Test effective positions computation."""
        # Equal weight across 4 assets
        w = np.array([0.25, 0.25, 0.25, 0.25])
        n_eff = compute_effective_positions(w)
        
        # Should be 4.0
        assert np.isclose(n_eff, 4.0, atol=1e-6)
        
        # Single asset
        w_single = np.array([1.0, 0.0, 0.0, 0.0])
        n_eff_single = compute_effective_positions(w_single)
        
        # Should be 1.0
        assert np.isclose(n_eff_single, 1.0, atol=1e-6)


class TestHMMUtils:
    """Test HMM utilities."""
    
    def test_hmm_wrapper_fit(self):
        """Test HMM fitting."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 500
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        
        # Fit HMM
        model = GaussianHMMWrapper(
            n_components=3,
            covariance_type='full',
            n_iter=50,
            tol=1e-4,
            n_init=2,
            random_state=42
        )
        model.fit(X)
        
        # Check fitted
        assert model.fitted
        assert model.n_components == 3
        
        # Check components
        components = model.get_components()
        assert len(components) == 3
        
        for mu, Sigma in components:
            assert mu.shape == (n_features,)
            assert Sigma.shape == (n_features, n_features)
            
            # Covariance should be symmetric
            assert np.allclose(Sigma, Sigma.T, atol=1e-6)
    
    def test_hmm_forward_probabilities(self):
        """Test forward probabilities computation."""
        np.random.seed(42)
        n_samples = 100
        n_features = 3
        
        X = np.random.randn(n_samples, n_features)
        
        model = GaussianHMMWrapper(
            n_components=2,
            covariance_type='full',
            n_iter=50,
            tol=1e-4,
            n_init=2,
            random_state=42
        )
        model.fit(X)
        
        # Compute forward probabilities
        alpha = model.compute_forward_probabilities(X)
        
        # Check shape
        assert alpha.shape == (n_samples, 2)
        
        # Each row should sum to 1
        row_sums = alpha.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
        
        # All probabilities should be in [0, 1]
        assert np.all(alpha >= 0)
        assert np.all(alpha <= 1)
    
    def test_hmm_predictive_probabilities(self):
        """Test predictive state probabilities."""
        np.random.seed(42)
        n_samples = 100
        n_features = 3
        
        X = np.random.randn(n_samples, n_features)
        
        model = GaussianHMMWrapper(
            n_components=2,
            covariance_type='full',
            n_iter=50,
            tol=1e-4,
            n_init=2,
            random_state=42
        )
        model.fit(X)
        
        # Compute predictive probabilities
        pred_probs = model.compute_predictive_probabilities(X)
        
        # Check shape
        assert pred_probs.shape == (n_samples, 2)
        
        # Each row should sum to 1
        row_sums = pred_probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)
        
        # All probabilities should be in [0, 1]
        assert np.all(pred_probs >= 0)
        assert np.all(pred_probs <= 1)


class TestPerformanceMetrics:
    """Test performance metrics computation."""
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio computation."""
        analyzer = PerformanceAnalyzer(annualization_factor=252)
        
        # Constant positive returns
        returns = np.ones(252) * 0.001  # 0.1% daily
        
        sharpe = analyzer.compute_sharpe_ratio(returns)
        
        # Sharpe should be very high (no volatility in this simplified case)
        # Actually, std will be 0, so we need varying returns
        returns = np.random.randn(252) * 0.01 + 0.001
        sharpe = analyzer.compute_sharpe_ratio(returns)
        
        # Should be positive
        assert sharpe > 0
    
    def test_sortino_ratio(self):
        """Test Sortino ratio computation."""
        analyzer = PerformanceAnalyzer(annualization_factor=252)
        
        # Mix of positive and negative returns
        returns = np.random.randn(252) * 0.01 + 0.001
        
        sortino = analyzer.compute_sortino_ratio(returns)
        
        # Should be positive (positive mean return)
        assert sortino > 0
    
    def test_sortino_no_downside(self):
        """Test Sortino with no negative returns."""
        analyzer = PerformanceAnalyzer(annualization_factor=252)
        
        # All positive returns
        returns = np.abs(np.random.randn(252) * 0.01) + 0.001
        
        sortino = analyzer.compute_sortino_ratio(returns)
        
        # Should be infinite
        assert sortino == np.inf
    
    def test_max_drawdown(self):
        """Test maximum drawdown computation."""
        analyzer = PerformanceAnalyzer(annualization_factor=252)
        
        # Create returns with known drawdown
        returns = np.array([0.01, 0.01, -0.02, -0.01, 0.005])
        
        max_dd = analyzer.compute_max_drawdown(returns)
        
        # Should be negative
        assert max_dd < 0
    
    def test_max_drawdown_no_loss(self):
        """Test max drawdown with no losses."""
        analyzer = PerformanceAnalyzer(annualization_factor=252)
        
        # All positive returns
        returns = np.ones(100) * 0.01
        
        max_dd = analyzer.compute_max_drawdown(returns)
        
        # Should be zero (no drawdown)
        assert np.isclose(max_dd, 0.0, atol=1e-10)
    
    def test_compute_all_metrics(self):
        """Test computation of all metrics."""
        analyzer = PerformanceAnalyzer(annualization_factor=252)
        
        # Create sample results
        np.random.seed(42)
        n_days = 500
        
        results_df = pd.DataFrame({
            'r_port': np.random.randn(n_days) * 0.01 + 0.0003,
            'turnover': np.random.rand(n_days) * 0.1,
            'n_eff': np.random.rand(n_days) * 3 + 2
        })
        
        metrics = analyzer.compute_all_metrics(results_df)
        
        # Check all expected keys present
        expected_keys = [
            'sharpe', 'sortino', 'max_drawdown', 'avg_turnover',
            'pct95_turnover', 'pct_turnover_gt_1pct', 'pct_turnover_gt_5pct',
            'avg_n_eff', 'median_n_eff', 'avg_return', 'std_return', 'total_return'
        ]
        
        for key in expected_keys:
            assert key in metrics
        
        # Check ranges
        assert metrics['avg_turnover'] >= 0
        assert metrics['avg_n_eff'] >= 1
        assert 0 <= metrics['pct_turnover_gt_1pct'] <= 100


class TestStrictCausality:
    """Test strict causality enforcement in feature construction."""
    
    def test_feature_lag(self):
        """Test that decision features are properly lagged."""
        # Create mock price series
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.DataFrame({
            'SPX': np.exp(np.cumsum(np.random.randn(100) * 0.01)),
            'BOND': np.exp(np.cumsum(np.random.randn(100) * 0.005))
        }, index=dates)
        
        # Compute returns
        returns = np.log(prices).diff().dropna()
        
        # Compute rolling features
        sigma = returns.rolling(window=20, min_periods=20).std()
        features = sigma.dropna()
        
        # Create decision features (shifted by 1)
        features_decision = features.shift(1).dropna()
        
        # At day t, decision features should equal raw features at day t-1
        for i in range(1, len(features)):
            date_t = features.index[i]
            date_t_minus_1 = features.index[i-1]
            
            if date_t in features_decision.index:
                # features_decision[t] should equal features[t-1]
                assert np.allclose(
                    features_decision.loc[date_t].values,
                    features.loc[date_t_minus_1].values,
                    atol=1e-10
                )


class TestWeightConstraints:
    """Test portfolio weight constraints."""
    
    def test_weights_sum_to_one(self):
        """Test that optimized weights sum to 1."""
        optimizer = MVOOptimizer(gamma=1.0, tau=0.001, w_max=0.6)
        
        np.random.seed(42)
        n_tests = 10
        n_assets = 5
        
        for _ in range(n_tests):
            mu = np.random.randn(n_assets) * 0.001
            Sigma = np.eye(n_assets) * 0.01 + np.random.rand(n_assets, n_assets) * 0.001
            Sigma = (Sigma + Sigma.T) / 2  # Symmetrize
            Sigma = make_positive_definite(Sigma)
            
            w_prev = np.ones(n_assets) / n_assets
            w_opt = optimizer.optimize(mu, Sigma, w_prev)
            
            # Sum should be 1
            assert np.isclose(np.sum(w_opt), 1.0, atol=1e-4)
    
    def test_weights_nonnegative(self):
        """Test that weights are non-negative (long-only)."""
        optimizer = MVOOptimizer(gamma=1.0, tau=0.001, w_max=0.6)
        
        np.random.seed(42)
        n_assets = 5
        
        mu = np.random.randn(n_assets) * 0.001
        Sigma = np.eye(n_assets) * 0.01
        w_prev = np.ones(n_assets) / n_assets
        
        w_opt = optimizer.optimize(mu, Sigma, w_prev)
        
        # All weights should be >= 0
        assert np.all(w_opt >= -1e-6)
    
    def test_weights_bounded(self):
        """Test that weights respect maximum bound."""
        w_max = 0.6
        optimizer = MVOOptimizer(gamma=1.0, tau=0.001, w_max=w_max)
        
        np.random.seed(42)
        n_assets = 5
        
        mu = np.random.randn(n_assets) * 0.001
        mu[0] = 0.1  # Make one asset very attractive
        Sigma = np.eye(n_assets) * 0.01
        w_prev = np.ones(n_assets) / n_assets
        
        w_opt = optimizer.optimize(mu, Sigma, w_prev)
        
        # All weights should be <= w_max
        assert np.all(w_opt <= w_max + 1e-6)


def run_all_tests():
    """Run all tests and print results."""
    print("=" * 80)
    print("RUNNING COMPREHENSIVE TESTS")
    print("=" * 80)
    
    # Run pytest
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_all_tests()
