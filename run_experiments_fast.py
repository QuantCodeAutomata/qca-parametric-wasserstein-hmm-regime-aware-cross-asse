"""
Fast experiment runner with reduced computational requirements.
Uses simpler parameters for quick demonstration.
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
from performance_metrics import generate_all_visualizations, PerformanceAnalyzer


def main():
    """
    Run all experiments with faster settings.
    """
    # Set random seed
    np.random.seed(ExperimentConfig.RANDOM_SEED)
    
    # Override some config parameters for speed
    ExperimentConfig.ParametricHMM.HMM_MAX_ITER = 50  # Reduce from 200
    ExperimentConfig.ParametricHMM.HMM_N_INIT = 2     # Reduce from 5
    ExperimentConfig.ParametricHMM.MODEL_SELECTION_FREQ = 20  # Increase from 5
    
    # Save configuration
    ExperimentConfig.save_config()
    
    print("\n" + "=" * 80)
    print("CROSS-ASSET ALLOCATION EXPERIMENTS (FAST VERSION)")
    print("Parametric Wasserstein HMM vs KNN vs Benchmarks")
    print("=" * 80 + "\n")
    
    # STEP 1: Load and prepare data
    data = load_data()
    
    features_decision = data['features_decision']
    returns = data['returns']
    train_end_idx = data['train_end_idx']
    test_start_idx = data['test_start_idx']
    
    # STEP 2: Run Experiment 1 - Parametric Wasserstein HMM Strategy
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: PARAMETRIC WASSERSTEIN HMM STRATEGY")
    print("=" * 80 + "\n")
    
    parametric_strategy = ParametricWassersteinHMMStrategy()
    parametric_results = parametric_strategy.run_backtest(
        features_decision,
        returns,
        train_end_idx,
        test_start_idx
    )
    
    # STEP 3: Run Experiment 2 - KNN Strategy
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: KNN CONDITIONAL-MOMENT BASELINE")
    print("=" * 80 + "\n")
    
    knn_strategy = KNNStrategy(k_neighbors=ExperimentConfig.KNN.K_NEIGHBORS)
    knn_results = knn_strategy.run_backtest(
        features_decision,
        returns,
        train_end_idx,
        test_start_idx
    )
    
    # STEP 4: Run Experiment 3 - Benchmark Strategies
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: BENCHMARK STRATEGIES")
    print("=" * 80 + "\n")
    
    benchmark_strategies = BenchmarkStrategies()
    
    spx_results = benchmark_strategies.run_spx_buyhold(
        returns,
        test_start_idx
    )
    
    equalweight_results = benchmark_strategies.run_equalweight(
        returns,
        test_start_idx
    )
    
    # STEP 5: Aggregate results
    results_dict = {
        'Parametric HMM': parametric_results,
        'KNN': knn_results,
        'Equal-Weight': equalweight_results,
        'SPX B&H': spx_results
    }
    
    # STEP 6: Generate performance analysis and visualizations
    print("\n" + "=" * 80)
    print("GENERATING PERFORMANCE ANALYSIS AND VISUALIZATIONS")
    print("=" * 80 + "\n")
    
    generate_all_visualizations(
        results_dict,
        ExperimentConfig.ASSET_NAMES,
        output_dir='results'
    )
    
    # STEP 7: Save detailed results
    print("\nSaving detailed results...")
    for strategy_name, results_df in results_dict.items():
        filename = f"results_{strategy_name.replace(' ', '_').lower()}.csv"
        results_df.to_csv(f'results/{filename}')
        print(f"  Saved {filename}")
    
    # STEP 8: Generate markdown report
    from run_experiments import generate_markdown_report
    generate_markdown_report(results_dict)
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("=" * 80 + "\n")
    print("Results saved to results/ directory")
    print("  - RESULTS.md: Summary report")
    print("  - comparison_table.csv: Metrics comparison")
    print("  - *.png: Visualization plots")
    print("  - results_*.csv: Detailed daily results")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
