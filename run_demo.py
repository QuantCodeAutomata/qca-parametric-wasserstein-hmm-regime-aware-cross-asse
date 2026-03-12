"""
Minimal demonstration runner for quick validation.
Uses last 6 months of OOS period for demonstration.
"""
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from config import ExperimentConfig
from data_loader import load_data
from parametric_strategy import ParametricWassersteinHMMStrategy
from knn_strategy import KNNStrategy
from benchmark_strategies import BenchmarkStrategies
from performance_metrics import generate_all_visualizations


def main():
    """
    Run minimal demonstration of all strategies.
    """
    np.random.seed(42)
    
    # Speed optimizations
    ExperimentConfig.ParametricHMM.HMM_MAX_ITER = 30
    ExperimentConfig.ParametricHMM.HMM_N_INIT = 1
    ExperimentConfig.ParametricHMM.MODEL_SELECTION_FREQ = 30
    ExperimentConfig.save_config()
    
    print("\n" + "=" * 80)
    print("MINIMAL DEMONSTRATION - LAST 6 MONTHS OOS")
    print("=" * 80 + "\n")
    
    # Load data
    data = load_data()
    features_decision = data['features_decision']
    returns = data['returns']
    train_end_idx = data['train_end_idx']
    test_start_idx = data['test_start_idx']
    
    # Use only last 6 months (~120 trading days) of OOS period
    demo_start_idx = len(features_decision) - 120
    if demo_start_idx < test_start_idx:
        demo_start_idx = test_start_idx
    
    print(f"\nUsing demo period: last 120 days")
    print(f"Demo start index: {demo_start_idx} / {len(features_decision)}")
    
    # Run strategies on demo period
    print("\n" + "=" * 80)
    print("RUNNING PARAMETRIC HMM...")
    print("=" * 80)
    parametric_strategy = ParametricWassersteinHMMStrategy()
    parametric_results = parametric_strategy.run_backtest(
        features_decision, returns, train_end_idx, demo_start_idx
    )
    
    print("\n" + "=" * 80)
    print("RUNNING KNN...")
    print("=" * 80)
    knn_strategy = KNNStrategy(k_neighbors=50)
    knn_results = knn_strategy.run_backtest(
        features_decision, returns, train_end_idx, demo_start_idx
    )
    
    print("\n" + "=" * 80)
    print("RUNNING BENCHMARKS...")
    print("=" * 80)
    benchmark_strategies = BenchmarkStrategies()
    spx_results = benchmark_strategies.run_spx_buyhold(returns, demo_start_idx)
    equalweight_results = benchmark_strategies.run_equalweight(returns, demo_start_idx)
    
    # Aggregate
    results_dict = {
        'Parametric HMM': parametric_results,
        'KNN': knn_results,
        'Equal-Weight': equalweight_results,
        'SPX B&H': spx_results
    }
    
    # Generate outputs
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    generate_all_visualizations(results_dict, ExperimentConfig.ASSET_NAMES, 'results')
    
    # Save results
    for name, df in results_dict.items():
        filename = f"results_{name.replace(' ', '_').lower()}.csv"
        df.to_csv(f'results/{filename}')
    
    from run_experiments import generate_markdown_report
    generate_markdown_report(results_dict)
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE! Results saved to results/")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
