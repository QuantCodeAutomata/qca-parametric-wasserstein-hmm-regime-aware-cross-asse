# Parametric Wasserstein HMM Regime-Aware Cross-Asset Allocation

This repository implements a comprehensive quantitative finance research study comparing different cross-asset allocation strategies, with a focus on parametric Wasserstein HMM-based regime inference.

## Overview

The project implements and compares three distinct approaches to daily cross-asset portfolio allocation across 5 assets (SPX, BOND, GOLD, OIL, USD):

1. **Parametric Wasserstein HMM Strategy** (exp_1): Advanced regime-aware allocation using Hidden Markov Models with Wasserstein distance-based template tracking
2. **KNN Conditional-Moment Baseline** (exp_2): Non-parametric approach using K-nearest neighbors for regime inference
3. **Benchmark Strategies** (exp_3): SPX buy-and-hold and equal-weight portfolio baselines

## Key Features

- **Strict Causal Feature Engineering**: All features for decision at day t use only data through t-1
- **Predictive Model Order Selection**: Dynamic HMM state selection with validation-based scoring
- **Wasserstein Geometry**: Template identity preservation through rolling re-estimation
- **Transaction-Cost-Aware Optimization**: Mean-variance optimization with L1 turnover penalty
- **Comprehensive Performance Analysis**: Sharpe ratio, Sortino ratio, maximum drawdown, turnover metrics

## Repository Structure

```
.
├── config.py                    # Central configuration for all experiments
├── data_loader.py              # Data loading and feature construction
├── wasserstein_utils.py        # Wasserstein distance computation utilities
├── hmm_utils.py                # HMM fitting and model order selection
├── mvo_optimizer.py            # Mean-variance optimization with turnover penalty
├── parametric_strategy.py      # Parametric Wasserstein HMM strategy
├── knn_strategy.py             # KNN conditional-moment baseline
├── benchmark_strategies.py     # SPX buy-and-hold and equal-weight benchmarks
├── performance_metrics.py      # Performance computation and visualization
├── run_experiments.py          # Main experiment runner
├── test_experiments.py         # Comprehensive test suite
└── results/                    # Output directory for results and plots
    ├── RESULTS.md              # Summary report
    ├── comparison_table.csv    # Metrics comparison
    ├── *.png                   # Visualization plots
    └── results_*.csv           # Detailed daily results
```

## Installation

### Requirements

- Python 3.8+
- See installed libraries in container (numpy, pandas, scipy, scikit-learn, etc.)

### Setup

```bash
# Clone repository
git clone https://github.com/QuantCodeAutomata/qca-parametric-wasserstein-hmm-regime-aware-cross-asse.git
cd qca-parametric-wasserstein-hmm-regime-aware-cross-asse

# Install dependencies (if needed)
pip install -r requirements.txt
```

## Usage

### Run All Experiments

```bash
python run_experiments.py
```

This will:
1. Download and prepare data from Yahoo Finance (2005-2026)
2. Run all three experiments
3. Generate performance metrics and visualizations
4. Save results to `results/` directory

### Run Tests

```bash
python test_experiments.py
```

Or with pytest:

```bash
pytest test_experiments.py -v
```

## Methodology

### Data Preparation

- **Assets**: SPX (^GSPC), BOND (TLT), GOLD (GLD), OIL (USO), USD (UUP)
- **Period**: 2005-01-01 to 2026-02-28
- **Train/Test Split**: ~2023-05-01
- **Features**: Log returns, 60-day rolling volatility, 20-day rolling mean
- **Strict Causality**: Decision at day t uses features from t-1

### Parametric Wasserstein HMM Strategy

1. **Template Initialization**: Fit G=6 state HMM on calibration window
2. **Model Order Selection**: Every 5 days, select optimal K from {2,3,4,5,6,7,8} using predictive validation
3. **HMM Fitting**: Fit K-state Gaussian HMM on expanding history
4. **Template Mapping**: Map HMM components to templates via 2-Wasserstein distance
5. **Template Update**: Exponential smoothing with eta=0.05
6. **Moment Aggregation**: Compute template-weighted expected returns and covariance
7. **MVO Optimization**: Transaction-cost-aware mean-variance optimization

### KNN Baseline Strategy

1. **Neighbor Retrieval**: Find K=50 nearest neighbors using Euclidean distance
2. **Moment Estimation**: Average neighbor returns, Ledoit-Wolf shrinkage covariance
3. **MVO Optimization**: Identical optimization layer as parametric strategy

### Mean-Variance Optimization

Solve for all strategies:
```
max_w  mu^T w - gamma * w^T Sigma w - tau * ||w - w_prev||_1
s.t.   sum(w) = 1
       w >= 0
       w <= 0.6
```

Parameters: gamma=1.0, tau=0.001, w_max=0.6

## Expected Results

### Performance Targets (OOS: ~2023-05 to ~2026-02)

| Strategy | Sharpe | Sortino | Max DD | Avg TO | 95% TO |
|----------|--------|---------|--------|--------|--------|
| Parametric HMM | ~2.18 | ~2.82 | ~-5.43% | ~0.0079 | ~0.0504 |
| KNN | ~1.81 | - | ~-12.52% | ~0.5665 | ~1.0000 |
| Equal-Weight | ~1.59 | ~2.27 | ~-9.87% | 0 | 0 |
| SPX B&H | ~1.18 | ~1.50 | ~-14.62% | 0 | 0 |

### Key Findings

- **Parametric HMM achieves highest Sharpe ratio** (~2.18 vs ~1.81 for KNN)
- **Dramatically lower turnover** (~0.0079 vs ~0.5665 for KNN)
- **Smallest maximum drawdown** (-5.43% vs -12.52% for KNN)
- **Smooth weight evolution** vs frequent reallocations in KNN
- **Template identity preservation** through Wasserstein geometry

## Configuration

All parameters are centrally configured in `config.py`:

- **Tickers**: Yahoo Finance symbols for 5 assets
- **Date Ranges**: Data period and train/test split
- **Feature Windows**: 60-day volatility, 20-day mean
- **HMM Parameters**: Number of templates, model selection frequency, convergence criteria
- **KNN Parameters**: Number of neighbors, distance metric
- **MVO Parameters**: Risk aversion, turnover penalty, position limits

## Testing

Comprehensive test suite validates:

- Wasserstein distance computation
- Matrix operations (symmetrization, square root, positive definiteness)
- HMM fitting and forward algorithm
- MVO optimization (constraints, convergence)
- Performance metrics computation
- Strict causality enforcement
- Weight constraints (sum to 1, long-only, position limits)

## Visualizations

The experiment generates:

- **Cumulative Returns**: Comparison across all strategies
- **Drawdown Curves**: Time-varying drawdown visualization
- **Weight Evolution**: Stacked area charts of portfolio weights
- **Turnover Comparison**: Daily turnover time series

## Citation

If you use this code for research, please cite:

```
Parametric Wasserstein HMM Regime-Aware Cross-Asset Allocation
QuantCodeAutomata, 2026
https://github.com/QuantCodeAutomata/qca-parametric-wasserstein-hmm-regime-aware-cross-asse
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
