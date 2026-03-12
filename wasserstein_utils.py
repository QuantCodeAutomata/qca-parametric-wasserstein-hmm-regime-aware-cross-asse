"""
Wasserstein distance utilities for Gaussian distributions.
Implements 2-Wasserstein distance computation between multivariate Gaussians.
"""
import numpy as np
from scipy import linalg
from typing import Tuple


def symmetrize_matrix(A: np.ndarray) -> np.ndarray:
    """
    Symmetrize a matrix.
    
    Args:
        A: Input matrix
        
    Returns:
        Symmetrized matrix (A + A^T) / 2
    """
    return (A + A.T) / 2


def make_positive_definite(A: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """
    Ensure matrix is positive definite by adding diagonal jitter if needed.
    
    Args:
        A: Input covariance matrix
        jitter: Minimum diagonal jitter to add if not positive definite
        
    Returns:
        Positive definite matrix
    """
    # Symmetrize first
    A_sym = symmetrize_matrix(A)
    
    # Check if positive definite
    try:
        np.linalg.cholesky(A_sym)
        return A_sym
    except np.linalg.LinAlgError:
        # Find minimum eigenvalue and add sufficient jitter
        eigvals = np.linalg.eigvalsh(A_sym)
        min_eigval = np.min(eigvals)
        
        if min_eigval < jitter:
            # Add enough to make smallest eigenvalue at least jitter
            jitter_needed = jitter - min_eigval + jitter
        else:
            jitter_needed = jitter
        
        return A_sym + jitter_needed * np.eye(A_sym.shape[0])


def matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """
    Compute matrix square root via eigendecomposition.
    
    Args:
        A: Positive semi-definite matrix
        
    Returns:
        Matrix square root A^{1/2}
    """
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(A)
    
    # Handle negative eigenvalues (numerical errors)
    eigvals = np.maximum(eigvals, 0)
    
    # Compute square root
    sqrt_eigvals = np.sqrt(eigvals)
    A_sqrt = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    
    # Symmetrize to remove numerical asymmetry
    A_sqrt = symmetrize_matrix(A_sqrt)
    
    return A_sqrt


def wasserstein2_squared(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray
) -> float:
    """
    Compute squared 2-Wasserstein distance between two Gaussian distributions.
    
    The 2-Wasserstein distance between N(mu1, Sigma1) and N(mu2, Sigma2) is:
    W_2^2 = ||mu1 - mu2||_2^2 + Tr(Sigma1 + Sigma2 - 2*(Sigma2^{1/2} Sigma1 Sigma2^{1/2})^{1/2})
    
    Args:
        mu1: Mean of first Gaussian (d-dimensional)
        Sigma1: Covariance of first Gaussian (d x d)
        mu2: Mean of second Gaussian (d-dimensional)
        Sigma2: Covariance of second Gaussian (d x d)
        
    Returns:
        Squared 2-Wasserstein distance W_2^2(N(mu1, Sigma1), N(mu2, Sigma2))
    """
    # Mean distance term
    mean_dist_sq = np.sum((mu1 - mu2) ** 2)
    
    # Covariance term: Tr(Sigma1 + Sigma2 - 2*(Sigma2^{1/2} Sigma1 Sigma2^{1/2})^{1/2})
    
    # Compute Sigma2^{1/2}
    Sigma2_sqrt = matrix_sqrt(Sigma2)
    
    # Compute M = Sigma2^{1/2} @ Sigma1 @ Sigma2^{1/2}
    M = Sigma2_sqrt @ Sigma1 @ Sigma2_sqrt
    
    # Compute M^{1/2}
    M_sqrt = matrix_sqrt(M)
    
    # Covariance distance
    cov_term = np.trace(Sigma1) + np.trace(Sigma2) - 2 * np.trace(M_sqrt)
    
    # Total W2^2 (take real part to handle numerical errors)
    w2_squared = mean_dist_sq + cov_term
    
    # Ensure non-negative (numerical errors can make it slightly negative)
    w2_squared = max(0.0, np.real(w2_squared))
    
    return w2_squared


def wasserstein2(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray
) -> float:
    """
    Compute 2-Wasserstein distance between two Gaussian distributions.
    
    Args:
        mu1: Mean of first Gaussian
        Sigma1: Covariance of first Gaussian
        mu2: Mean of second Gaussian
        Sigma2: Covariance of second Gaussian
        
    Returns:
        2-Wasserstein distance W_2(N(mu1, Sigma1), N(mu2, Sigma2))
    """
    return np.sqrt(wasserstein2_squared(mu1, Sigma1, mu2, Sigma2))


def compute_wasserstein_distance_matrix(
    components: list,
    templates: list
) -> np.ndarray:
    """
    Compute pairwise Wasserstein distances between HMM components and templates.
    
    Args:
        components: List of (mu, Sigma) tuples for HMM components
        templates: List of (mu, Sigma) tuples for templates
        
    Returns:
        Distance matrix of shape (n_components, n_templates)
    """
    n_components = len(components)
    n_templates = len(templates)
    
    dist_matrix = np.zeros((n_components, n_templates))
    
    for k in range(n_components):
        mu_k, Sigma_k = components[k]
        for g in range(n_templates):
            mu_g, Sigma_g = templates[g]
            dist_matrix[k, g] = wasserstein2_squared(mu_g, Sigma_g, mu_k, Sigma_k)
    
    return dist_matrix


def map_components_to_templates(
    components: list,
    templates: list
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map HMM components to templates via Wasserstein distance.
    
    For each component k, assigns it to template g(k) = argmin_g W_2^2(template_g, component_k).
    
    Args:
        components: List of (mu, Sigma) tuples for HMM components
        templates: List of (mu, Sigma) tuples for templates
        
    Returns:
        Tuple of (mapping, distances) where:
        - mapping: array of shape (n_components,) with template indices
        - distances: array of shape (n_components,) with minimum distances
    """
    # Compute distance matrix
    dist_matrix = compute_wasserstein_distance_matrix(components, templates)
    
    # Find argmin for each component
    mapping = np.argmin(dist_matrix, axis=1)
    distances = np.min(dist_matrix, axis=1)
    
    return mapping, distances
