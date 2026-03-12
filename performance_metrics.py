"""
Performance metrics and visualization utilities.
Computes Sharpe ratio, Sortino ratio, maximum drawdown, and other metrics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import os

sns.set_style('whitegrid')


class PerformanceAnalyzer:
    """
    Compute performance metrics and generate visualizations.
    """
    
    def __init__(self, annualization_factor: int = 252):
        """
        Initialize performance analyzer.
        
        Args:
            annualization_factor: Number of trading days per year
        """
        self.annualization_factor = annualization_factor
    
    def compute_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Compute annualized Sharpe ratio.
        
        Args:
            returns: Daily returns
            
        Returns:
            Annualized Sharpe ratio
        """
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        
        if std_ret == 0:
            return 0.0
        
        sharpe = (mean_ret * self.annualization_factor) / (std_ret * np.sqrt(self.annualization_factor))
        return sharpe
    
    def compute_sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Compute annualized Sortino ratio.
        
        Args:
            returns: Daily returns
            
        Returns:
            Annualized Sortino ratio
        """
        mean_ret = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return np.inf
        
        sortino = (mean_ret * self.annualization_factor) / (downside_std * np.sqrt(self.annualization_factor))
        return sortino
    
    def compute_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Compute maximum drawdown from log returns.
        
        Args:
            returns: Daily log returns
            
        Returns:
            Maximum drawdown (negative value)
        """
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_dd = np.min(drawdown)
        return max_dd
    
    def compute_all_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute all performance metrics.
        
        Args:
            results_df: DataFrame with 'r_port', 'turnover', 'n_eff' columns
            
        Returns:
            Dictionary of metrics
        """
        returns = results_df['r_port'].values
        turnover = results_df['turnover'].values
        n_eff = results_df['n_eff'].values
        
        metrics = {
            'sharpe': self.compute_sharpe_ratio(returns),
            'sortino': self.compute_sortino_ratio(returns),
            'max_drawdown': self.compute_max_drawdown(returns),
            'avg_turnover': np.mean(turnover),
            'pct95_turnover': np.percentile(turnover, 95),
            'pct_turnover_gt_1pct': np.mean(turnover > 0.01) * 100,
            'pct_turnover_gt_5pct': np.mean(turnover > 0.05) * 100,
            'avg_n_eff': np.mean(n_eff),
            'median_n_eff': np.median(n_eff),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns, ddof=1),
            'total_return': np.sum(returns)
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], strategy_name: str) -> None:
        """
        Print metrics in formatted table.
        
        Args:
            metrics: Dictionary of metrics
            strategy_name: Name of strategy
        """
        print(f"\n{strategy_name} Performance Metrics:")
        print("-" * 60)
        print(f"  Annualized Sharpe Ratio:       {metrics['sharpe']:>10.4f}")
        print(f"  Annualized Sortino Ratio:      {metrics['sortino']:>10.4f}")
        print(f"  Maximum Drawdown:              {metrics['max_drawdown']:>10.4f} ({metrics['max_drawdown']*100:>6.2f}%)")
        print(f"  Average Daily Turnover:        {metrics['avg_turnover']:>10.4f}")
        print(f"  95th %ile Daily Turnover:      {metrics['pct95_turnover']:>10.4f}")
        print(f"  % Days Turnover > 1%:          {metrics['pct_turnover_gt_1pct']:>10.2f}%")
        print(f"  % Days Turnover > 5%:          {metrics['pct_turnover_gt_5pct']:>10.2f}%")
        print(f"  Average Effective Positions:   {metrics['avg_n_eff']:>10.4f}")
        print(f"  Median Effective Positions:    {metrics['median_n_eff']:>10.4f}")
        print("-" * 60)
    
    def plot_cumulative_returns(
        self,
        results_dict: Dict[str, pd.DataFrame],
        save_path: str = None
    ) -> None:
        """
        Plot cumulative log returns for all strategies.
        
        Args:
            results_dict: Dictionary mapping strategy name to results DataFrame
            save_path: Path to save figure
        """
        plt.figure(figsize=(14, 7))
        
        for strategy_name, results_df in results_dict.items():
            cum_returns = np.cumsum(results_df['r_port'].values)
            plt.plot(results_df.index, cum_returns, label=strategy_name, linewidth=2)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Log Return', fontsize=12)
        plt.title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved cumulative returns plot to {save_path}")
        
        plt.close()
    
    def plot_drawdowns(
        self,
        results_dict: Dict[str, pd.DataFrame],
        save_path: str = None
    ) -> None:
        """
        Plot drawdown curves for all strategies.
        
        Args:
            results_dict: Dictionary mapping strategy name to results DataFrame
            save_path: Path to save figure
        """
        plt.figure(figsize=(14, 7))
        
        for strategy_name, results_df in results_dict.items():
            cum_returns = np.cumsum(results_df['r_port'].values)
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = cum_returns - running_max
            plt.plot(results_df.index, drawdown * 100, label=strategy_name, linewidth=2)
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.title('Drawdown Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved drawdown plot to {save_path}")
        
        plt.close()
    
    def plot_weights_stacked(
        self,
        results_df: pd.DataFrame,
        asset_names: list,
        strategy_name: str,
        save_path: str = None
    ) -> None:
        """
        Plot stacked area chart of portfolio weights over time.
        
        Args:
            results_df: Results DataFrame with weight columns
            asset_names: List of asset names
            strategy_name: Name of strategy
            save_path: Path to save figure
        """
        plt.figure(figsize=(14, 7))
        
        # Extract weights
        weights = np.array([results_df[f'w_{asset}'].values for asset in asset_names]).T
        
        # Create stacked area plot
        plt.stackplot(
            results_df.index,
            *[weights[:, i] for i in range(len(asset_names))],
            labels=asset_names,
            alpha=0.8
        )
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Weight', fontsize=12)
        plt.title(f'{strategy_name} - Portfolio Weights Over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved weights plot to {save_path}")
        
        plt.close()
    
    def plot_turnover_timeseries(
        self,
        results_dict: Dict[str, pd.DataFrame],
        save_path: str = None
    ) -> None:
        """
        Plot turnover time series for comparison.
        
        Args:
            results_dict: Dictionary mapping strategy name to results DataFrame
            save_path: Path to save figure
        """
        plt.figure(figsize=(14, 7))
        
        for strategy_name, results_df in results_dict.items():
            plt.plot(
                results_df.index,
                results_df['turnover'].values * 100,
                label=strategy_name,
                linewidth=1.5,
                alpha=0.7
            )
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Daily Turnover (%)', fontsize=12)
        plt.title('Daily Turnover Comparison', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved turnover plot to {save_path}")
        
        plt.close()
    
    def create_comparison_table(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: str = None
    ) -> pd.DataFrame:
        """
        Create comparison table of all strategies.
        
        Args:
            metrics_dict: Dictionary mapping strategy name to metrics dict
            save_path: Path to save CSV
            
        Returns:
            DataFrame with comparison table
        """
        # Extract key metrics
        comparison_data = []
        for strategy_name, metrics in metrics_dict.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Sharpe': metrics['sharpe'],
                'Sortino': metrics['sortino'],
                'Max DD (%)': metrics['max_drawdown'] * 100,
                'Avg Turnover': metrics['avg_turnover'],
                '95% Turnover': metrics['pct95_turnover'],
                'Avg N_eff': metrics['avg_n_eff']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('Strategy')
        
        if save_path:
            comparison_df.to_csv(save_path)
            print(f"Saved comparison table to {save_path}")
        
        return comparison_df


def generate_all_visualizations(
    results_dict: Dict[str, pd.DataFrame],
    asset_names: list,
    output_dir: str = 'results'
) -> None:
    """
    Generate all visualizations and save to output directory.
    
    Args:
        results_dict: Dictionary mapping strategy name to results DataFrame
        asset_names: List of asset names
        output_dir: Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    analyzer = PerformanceAnalyzer()
    
    # Compute metrics for all strategies
    metrics_dict = {}
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    for strategy_name, results_df in results_dict.items():
        metrics = analyzer.compute_all_metrics(results_df)
        metrics_dict[strategy_name] = metrics
        analyzer.print_metrics(metrics, strategy_name)
    
    # Generate comparison table
    comparison_df = analyzer.create_comparison_table(
        metrics_dict,
        save_path=os.path.join(output_dir, 'comparison_table.csv')
    )
    print(f"\nComparison Table:")
    print(comparison_df.to_string())
    
    # Generate plots
    analyzer.plot_cumulative_returns(
        results_dict,
        save_path=os.path.join(output_dir, 'cumulative_returns.png')
    )
    
    analyzer.plot_drawdowns(
        results_dict,
        save_path=os.path.join(output_dir, 'drawdowns.png')
    )
    
    analyzer.plot_turnover_timeseries(
        results_dict,
        save_path=os.path.join(output_dir, 'turnover_comparison.png')
    )
    
    # Individual weight plots
    for strategy_name, results_df in results_dict.items():
        if strategy_name not in ['SPX B&H', 'Equal-Weight']:  # Skip static benchmarks
            analyzer.plot_weights_stacked(
                results_df,
                asset_names,
                strategy_name,
                save_path=os.path.join(output_dir, f'weights_{strategy_name.replace(" ", "_").lower()}.png')
            )
    
    print("\n" + "=" * 80)
    print(f"All visualizations saved to {output_dir}/")
    print("=" * 80)
