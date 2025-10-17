#!/usr/bin/env python3
"""
Plot Functional Metrics
~~~~~~~~~~~~~~~~~~~~~~~

Script to visualize metrics from the functional metrics output files:
- model_cluster_scores.json
- cluster_scores.json  
- model_scores.json

Creates comprehensive visualizations for proportion, quality, quality_delta, 
proportion_delta, and size metrics with confidence intervals when available.
"""

import argparse
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings
import wandb
# import weave

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

def load_metrics_files(results_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load the three metrics JSON files."""
    model_cluster_path = results_dir / "model_cluster_scores.json"
    cluster_path = results_dir / "cluster_scores.json"
    model_path = results_dir / "model_scores.json"
    
    if not all(path.exists() for path in [model_cluster_path, cluster_path, model_path]):
        missing = [p.name for p in [model_cluster_path, cluster_path, model_path] if not p.exists()]
        raise FileNotFoundError(f"Missing files in {results_dir}: {missing}")
    
    with open(model_cluster_path) as f:
        model_cluster_scores = json.load(f)
    with open(cluster_path) as f:
        cluster_scores = json.load(f)
    with open(model_path) as f:
        model_scores = json.load(f)
    
    return model_cluster_scores, cluster_scores, model_scores

def create_model_cluster_dataframe(model_cluster_scores: Dict[str, Any]) -> pd.DataFrame:
    """Convert model-cluster scores to a tidy dataframe."""
    rows = []
    for model, clusters in model_cluster_scores.items():
        for cluster, metrics in clusters.items():
            row = {
                'model': model,
                'cluster': cluster,
                'size': metrics.get('size', 0),
                'proportion': metrics.get('proportion', 0),
                'proportion_delta': metrics.get('proportion_delta', 0)
            }
            
            # Add confidence intervals if available
            if 'proportion_ci' in metrics:
                ci = metrics['proportion_ci']
                row.update({
                    'proportion_ci_lower': ci.get('lower', 0),
                    'proportion_ci_upper': ci.get('upper', 0),
                    'proportion_ci_mean': ci.get('mean', 0)
                })
            
            if 'proportion_delta_ci' in metrics:
                ci = metrics['proportion_delta_ci']
                row.update({
                    'proportion_delta_ci_lower': ci.get('lower', 0),
                    'proportion_delta_ci_upper': ci.get('upper', 0),
                    'proportion_delta_ci_mean': ci.get('mean', 0)
                })
            
            # Add significance flags
            row['proportion_delta_significant'] = metrics.get('proportion_delta_significant', False)
            
            # Add quality metrics
            quality = metrics.get('quality', {})
            quality_delta = metrics.get('quality_delta', {})
            quality_ci = metrics.get('quality_ci', {})
            quality_delta_ci = metrics.get('quality_delta_ci', {})
            quality_delta_significant = metrics.get('quality_delta_significant', {})
            
            for metric_name in quality.keys():
                row[f'quality_{metric_name}'] = quality[metric_name]
                row[f'quality_delta_{metric_name}'] = quality_delta.get(metric_name, 0)
                row[f'quality_delta_{metric_name}_significant'] = quality_delta_significant.get(metric_name, False)
                
                if metric_name in quality_ci:
                    ci = quality_ci[metric_name]
                    row.update({
                        f'quality_{metric_name}_ci_lower': ci.get('lower', 0),
                        f'quality_{metric_name}_ci_upper': ci.get('upper', 0),
                        f'quality_{metric_name}_ci_mean': ci.get('mean', 0)
                    })
                
                if metric_name in quality_delta_ci:
                    ci = quality_delta_ci[metric_name]
                    row.update({
                        f'quality_delta_{metric_name}_ci_lower': ci.get('lower', 0),
                        f'quality_delta_{metric_name}_ci_upper': ci.get('upper', 0),
                        f'quality_delta_{metric_name}_ci_mean': ci.get('mean', 0)
                    })
            
            rows.append(row)
    
    return pd.DataFrame(rows)

def create_cluster_dataframe(cluster_scores: Dict[str, Any]) -> pd.DataFrame:
    """Convert cluster scores to a tidy dataframe."""
    rows = []
    for cluster, metrics in cluster_scores.items():
        row = {
            'cluster': cluster,
            'size': metrics.get('size', 0),
            'proportion': metrics.get('proportion', 0)
        }
        
        # Add confidence intervals if available
        if 'proportion_ci' in metrics:
            ci = metrics['proportion_ci']
            row.update({
                'proportion_ci_lower': ci.get('lower', 0),
                'proportion_ci_upper': ci.get('upper', 0),
                'proportion_ci_mean': ci.get('mean', 0)
            })
        
        # Add quality metrics
        quality = metrics.get('quality', {})
        quality_delta = metrics.get('quality_delta', {})
        quality_ci = metrics.get('quality_ci', {})
        quality_delta_ci = metrics.get('quality_delta_ci', {})
        quality_delta_significant = metrics.get('quality_delta_significant', {})
        
        for metric_name in quality.keys():
            row[f'quality_{metric_name}'] = quality[metric_name]
            row[f'quality_delta_{metric_name}'] = quality_delta.get(metric_name, 0)
            row[f'quality_delta_{metric_name}_significant'] = quality_delta_significant.get(metric_name, False)
            
            if metric_name in quality_ci:
                ci = quality_ci[metric_name]
                row.update({
                    f'quality_{metric_name}_ci_lower': ci.get('lower', 0),
                    f'quality_{metric_name}_ci_upper': ci.get('upper', 0),
                    f'quality_{metric_name}_ci_mean': ci.get('mean', 0)
                })
            
            if metric_name in quality_delta_ci:
                ci = quality_delta_ci[metric_name]
                row.update({
                    f'quality_delta_{metric_name}_ci_lower': ci.get('lower', 0),
                    f'quality_delta_{metric_name}_ci_upper': ci.get('upper', 0),
                    f'quality_delta_{metric_name}_ci_mean': ci.get('mean', 0)
                })
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def create_model_dataframe(model_scores: Dict[str, Any]) -> pd.DataFrame:
    """Convert model scores to a tidy dataframe."""
    rows = []
    for model, metrics in model_scores.items():
        row = {
            'model': model,
            'size': metrics.get('size', 0),
            'proportion': metrics.get('proportion', 0)
        }
        
        # Add confidence intervals if available
        if 'proportion_ci' in metrics:
            ci = metrics['proportion_ci']
            row.update({
                'proportion_ci_lower': ci.get('lower', 0),
                'proportion_ci_upper': ci.get('upper', 0),
                'proportion_ci_mean': ci.get('mean', 0)
            })
        
        # Add quality metrics
        quality = metrics.get('quality', {})
        quality_delta = metrics.get('quality_delta', {})
        quality_ci = metrics.get('quality_ci', {})
        quality_delta_ci = metrics.get('quality_delta_ci', {})
        quality_delta_significant = metrics.get('quality_delta_significant', {})
        
        for metric_name in quality.keys():
            row[f'quality_{metric_name}'] = quality[metric_name]
            row[f'quality_delta_{metric_name}'] = quality_delta.get(metric_name, 0)
            row[f'quality_delta_{metric_name}_significant'] = quality_delta_significant.get(metric_name, False)
            
            if metric_name in quality_ci:
                ci = quality_ci[metric_name]
                row.update({
                    f'quality_{metric_name}_ci_lower': ci.get('lower', 0),
                    f'quality_{metric_name}_ci_upper': ci.get('upper', 0),
                    f'quality_{metric_name}_ci_mean': ci.get('mean', 0)
                })
            
            if metric_name in quality_delta_ci:
                ci = quality_delta_ci[metric_name]
                row.update({
                    f'quality_delta_{metric_name}_ci_lower': ci.get('lower', 0),
                    f'quality_delta_{metric_name}_ci_upper': ci.get('upper', 0),
                    f'quality_delta_{metric_name}_ci_mean': ci.get('mean', 0)
                })
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def get_quality_metrics(df: pd.DataFrame) -> List[str]:
    """Extract quality metric names from dataframe columns."""
    quality_cols = [col for col in df.columns if col.startswith('quality_') and not col.endswith(('_ci_lower', '_ci_upper', '_ci_mean', '_significant'))]
    return [col.replace('quality_', '') for col in quality_cols]

def plot_heatmap(df: pd.DataFrame, value_col: str, title: str, output_path: Path, 
                pivot_index: str = 'model', pivot_columns: str = 'cluster', 
                significant_col: Optional[str] = None, annot: bool = True, 
                wandb_key: Optional[str] = None):
    """Create a heatmap plot."""
    pivot_df = df.pivot(index=pivot_index, columns=pivot_columns, values=value_col)
    
    fig, ax = plt.subplots(figsize=(max(10, len(pivot_df.columns)), max(6, len(pivot_df.index))))
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=annot, fmt='.3f', cmap='RdBu_r', center=0 if 'delta' in value_col else None,
                cbar_kws={'shrink': 0.8}, ax=ax)
    
    # Add significance markers if available
    if significant_col and significant_col in df.columns:
        sig_pivot = df.pivot(index=pivot_index, columns=pivot_columns, values=significant_col)
        for i, model in enumerate(pivot_df.index):
            for j, cluster in enumerate(pivot_df.columns):
                if sig_pivot.loc[model, cluster]:
                    ax.text(j + 0.5, i + 0.8, '*', ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(pivot_columns.capitalize(), fontsize=12)
    ax.set_ylabel(pivot_index.capitalize(), fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if requested
    if wandb_key and wandb.run:
        wandb.log({wandb_key: wandb.Image(str(output_path))})
    
    plt.close()

def plot_bar_with_ci(df: pd.DataFrame, x_col: str, y_col: str, title: str, output_path: Path,
                    ci_lower_col: Optional[str] = None, ci_upper_col: Optional[str] = None,
                    significant_col: Optional[str] = None, figsize: tuple = (12, 6),
                    wandb_key: Optional[str] = None):
    """Create a bar plot with optional confidence intervals."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar plot
    bars = ax.bar(df[x_col], df[y_col], alpha=0.7)
    
    # Add error bars if CI data available
    if ci_lower_col and ci_upper_col and ci_lower_col in df.columns and ci_upper_col in df.columns:
        yerr_lower = df[y_col] - df[ci_lower_col]
        yerr_upper = df[ci_upper_col] - df[y_col]
        ax.errorbar(df[x_col], df[y_col], yerr=[yerr_lower, yerr_upper], 
                   fmt='none', color='black', capsize=3, alpha=0.7)
    
    # Mark significant values
    if significant_col and significant_col in df.columns:
        for i, (bar, is_sig) in enumerate(zip(bars, df[significant_col])):
            if is_sig:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (max(df[y_col]) - min(df[y_col])) * 0.02,
                       '*', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col.capitalize(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if requested
    if wandb_key and wandb.run:
        wandb.log({wandb_key: wandb.Image(str(output_path))})
    
    plt.close()

def plot_grouped_bar_by_hue(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, 
                           title: str, output_path: Path, figsize: tuple = (14, 8),
                           ci_lower_col: Optional[str] = None, ci_upper_col: Optional[str] = None,
                           wandb_key: Optional[str] = None):
    """Create a grouped bar chart with hue coloring and optional confidence intervals."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grouped bar plot using seaborn
    sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, alpha=0.8)
    
    # Add error bars if CI data available
    if ci_lower_col and ci_upper_col and ci_lower_col in df.columns and ci_upper_col in df.columns:
        # Get unique categories and hues
        x_categories = df[x_col].unique()
        hue_categories = df[hue_col].unique()
        
        # Calculate bar positions
        n_hues = len(hue_categories)
        bar_width = 0.8 / n_hues  # seaborn default bar group width is 0.8
        
        for i, x_cat in enumerate(x_categories):
            for j, hue_cat in enumerate(hue_categories):
                # Filter data for this specific combination
                mask = (df[x_col] == x_cat) & (df[hue_col] == hue_cat)
                if mask.any():
                    subset = df[mask].iloc[0]  # Should be exactly one row
                    
                    # Calculate x position for this bar
                    x_pos = i + (j - (n_hues - 1) / 2) * bar_width
                    
                    # Calculate error bar values
                    y_val = subset[y_col]
                    y_lower = subset[ci_lower_col] if not pd.isna(subset[ci_lower_col]) else y_val
                    y_upper = subset[ci_upper_col] if not pd.isna(subset[ci_upper_col]) else y_val
                    
                    yerr_lower = y_val - y_lower
                    yerr_upper = y_upper - y_val
                    
                    # Add error bar
                    ax.errorbar(x_pos, y_val, yerr=[[yerr_lower], [yerr_upper]], 
                               fmt='none', color='black', capsize=3, alpha=0.7, linewidth=1)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col.capitalize(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Position legend outside the plot area
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Log to wandb if requested
    if wandb_key and wandb.run:
        wandb.log({wandb_key: wandb.Image(str(output_path))})
    
    plt.close()

def create_quality_plots(df: pd.DataFrame, quality_metrics: List[str], output_dir: Path, prefix: str, log_to_wandb: bool = False):
    """Create plots for all quality metrics."""
    for metric in quality_metrics:
        # Quality scores
        quality_col = f'quality_{metric}'
        if quality_col in df.columns:
            if prefix == 'model_cluster':
                # Heatmap for model-cluster
                wandb_key = f'heatmaps/quality_{metric}' if log_to_wandb else None
                plot_heatmap(df, quality_col, f'Quality: {metric.title()}', 
                           output_dir / f'{prefix}_quality_{metric}_heatmap.png',
                           wandb_key=wandb_key)
            else:
                # Bar plot for cluster/model aggregates
                x_col = 'cluster' if prefix == 'cluster' else 'model'
                ci_lower = f'{quality_col}_ci_lower' if f'{quality_col}_ci_lower' in df.columns else None
                ci_upper = f'{quality_col}_ci_upper' if f'{quality_col}_ci_upper' in df.columns else None
                
                plot_bar_with_ci(df, x_col, quality_col, f'Quality: {metric.title()} ({prefix.title()})',
                               output_dir / f'{prefix}_quality_{metric}_bar.png',
                               ci_lower, ci_upper)
        
        # Quality delta scores
        quality_delta_col = f'quality_delta_{metric}'
        if quality_delta_col in df.columns:
            significant_col = f'{quality_delta_col}_significant'
            
            if prefix == 'model_cluster':
                # Heatmap for model-cluster
                wandb_key = f'heatmaps/quality_delta_{metric}' if log_to_wandb else None
                plot_heatmap(df, quality_delta_col, f'Quality Delta: {metric.title()}', 
                           output_dir / f'{prefix}_quality_delta_{metric}_heatmap.png',
                           significant_col=significant_col, wandb_key=wandb_key)
            else:
                # Bar plot for cluster/model aggregates
                x_col = 'cluster' if prefix == 'cluster' else 'model'
                ci_lower = f'{quality_delta_col}_ci_lower' if f'{quality_delta_col}_ci_lower' in df.columns else None
                ci_upper = f'{quality_delta_col}_ci_upper' if f'{quality_delta_col}_ci_upper' in df.columns else None
                
                plot_bar_with_ci(df, x_col, quality_delta_col, f'Quality Delta: {metric.title()} ({prefix.title()})',
                               output_dir / f'{prefix}_quality_delta_{metric}_bar.png',
                               ci_lower, ci_upper, significant_col)

def create_grouped_quality_plots(df: pd.DataFrame, quality_metrics: List[str], output_dir: Path):
    """Create grouped bar charts for quality metrics with model as hue."""
    for metric in quality_metrics:
        # Quality scores grouped by cluster with model as hue
        quality_col = f'quality_{metric}'
        if quality_col in df.columns:
            ci_lower = f'{quality_col}_ci_lower' if f'{quality_col}_ci_lower' in df.columns else None
            ci_upper = f'{quality_col}_ci_upper' if f'{quality_col}_ci_upper' in df.columns else None
            
            plot_grouped_bar_by_hue(df, 'cluster', quality_col, 'model',
                                   f'Quality {metric.title()} by Cluster (with CI)', 
                                   output_dir / f'quality_{metric}_per_cluster_grouped.png',
                                   ci_lower_col=ci_lower, ci_upper_col=ci_upper)
        
        # Quality delta scores grouped by cluster with model as hue
        quality_delta_col = f'quality_delta_{metric}'
        if quality_delta_col in df.columns:
            ci_lower = f'{quality_delta_col}_ci_lower' if f'{quality_delta_col}_ci_lower' in df.columns else None
            ci_upper = f'{quality_delta_col}_ci_upper' if f'{quality_delta_col}_ci_upper' in df.columns else None
            
            plot_grouped_bar_by_hue(df, 'cluster', quality_delta_col, 'model',
                                   f'Quality Delta {metric.title()} by Cluster (with CI)', 
                                   output_dir / f'quality_delta_{metric}_per_cluster_grouped.png',
                                   ci_lower_col=ci_lower, ci_upper_col=ci_upper)

def main():
    parser = argparse.ArgumentParser(description='Plot functional metrics from JSON output files')
    parser.add_argument('results_folder', type=str, help='Path to folder containing the three JSON metrics files')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for plots (defaults to results_folder/plots)')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for saved plots')
    parser.add_argument('--log-to-wandb', action='store_true', help='Log plots to wandb')
    parser.add_argument('--wandb-project', type=str, default='marcel-metrics', help='Wandb project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='Wandb run name (defaults to results folder name)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_folder)
    if not results_dir.exists():
        raise ValueError(f"Results folder does not exist: {results_dir}")
    
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if requested
    if args.log_to_wandb:
        run_name = args.wandb_run_name or results_dir.name
        wandb.init(project=args.wandb_project, name=run_name)
        print(f"Initialized wandb project: {args.wandb_project}, run: {run_name}")
    
    print(f"Loading metrics from {results_dir}...")
    model_cluster_scores, cluster_scores, model_scores = load_metrics_files(results_dir)
    
    print("Converting to dataframes...")
    model_cluster_df = create_model_cluster_dataframe(model_cluster_scores)
    cluster_df = create_cluster_dataframe(cluster_scores)
    model_df = create_model_dataframe(model_scores)
    
    # Get quality metrics
    quality_metrics = get_quality_metrics(model_cluster_df)
    print(f"Found quality metrics: {quality_metrics}")
    
    print(f"Creating plots in {output_dir}...")
    
    # =============================================================================
    # PER CLUSTER PLOTS (cluster counts, proportions, quality, quality delta) 
    # - Simple bar plots (model-agnostic, aggregated across all models)
    # =============================================================================
    
    # Cluster counts (total across all models)
    plot_bar_with_ci(cluster_df, 'cluster', 'size', 'Total Conversation Count per Cluster',
                    output_dir / 'per_cluster_counts.png',
                    wandb_key='per_cluster/counts' if args.log_to_wandb else None)
    
    # Cluster proportions (what fraction of all conversations are in each cluster)
    proportion_ci_lower = 'proportion_ci_lower' if 'proportion_ci_lower' in cluster_df.columns else None
    proportion_ci_upper = 'proportion_ci_upper' if 'proportion_ci_upper' in cluster_df.columns else None
    plot_bar_with_ci(cluster_df, 'cluster', 'proportion', 'Proportion of All Conversations per Cluster',
                    output_dir / 'per_cluster_proportions.png',
                    ci_lower_col=proportion_ci_lower, ci_upper_col=proportion_ci_upper,
                    wandb_key='per_cluster/proportions' if args.log_to_wandb else None)
    
    # Cluster quality scores (average across all models for each quality metric)
    for metric in quality_metrics:
        quality_col = f'quality_{metric}'
        if quality_col in cluster_df.columns:
            ci_lower = f'{quality_col}_ci_lower' if f'{quality_col}_ci_lower' in cluster_df.columns else None
            ci_upper = f'{quality_col}_ci_upper' if f'{quality_col}_ci_upper' in cluster_df.columns else None
            
            plot_bar_with_ci(cluster_df, 'cluster', quality_col, f'Average Quality {metric.title()} per Cluster',
                            output_dir / f'per_cluster_quality_{metric}.png',
                            ci_lower_col=ci_lower, ci_upper_col=ci_upper,
                            wandb_key=f'per_cluster/quality_{metric}' if args.log_to_wandb else None)
    
    # Cluster quality delta scores (how each cluster compares to overall average)
    for metric in quality_metrics:
        quality_delta_col = f'quality_delta_{metric}'
        if quality_delta_col in cluster_df.columns:
            ci_lower = f'{quality_delta_col}_ci_lower' if f'{quality_delta_col}_ci_lower' in cluster_df.columns else None
            ci_upper = f'{quality_delta_col}_ci_upper' if f'{quality_delta_col}_ci_upper' in cluster_df.columns else None
            significant_col = f'{quality_delta_col}_significant' if f'{quality_delta_col}_significant' in cluster_df.columns else None
            
            plot_bar_with_ci(cluster_df, 'cluster', quality_delta_col, f'Quality Delta {metric.title()} per Cluster',
                            output_dir / f'per_cluster_quality_delta_{metric}.png',
                            ci_lower_col=ci_lower, ci_upper_col=ci_upper,
                            significant_col=significant_col,
                            wandb_key=f'per_cluster/quality_delta_{metric}' if args.log_to_wandb else None)
    
    # =============================================================================
    # PER MODEL PLOTS (model counts, quality)
    # - Simple bar plots
    # =============================================================================
    
    # Model quality scores (for each quality metric)  
    for metric in quality_metrics:
        quality_col = f'quality_{metric}'
        if quality_col in model_df.columns:
            ci_lower = f'{quality_col}_ci_lower' if f'{quality_col}_ci_lower' in model_df.columns else None
            ci_upper = f'{quality_col}_ci_upper' if f'{quality_col}_ci_upper' in model_df.columns else None
            
            plot_bar_with_ci(model_df, 'model', quality_col, f'Quality {metric.title()} by Model',
                            output_dir / f'per_model_quality_{metric}.png',
                            ci_lower_col=ci_lower, ci_upper_col=ci_upper,
                            wandb_key=f'per_model/quality_{metric}' if args.log_to_wandb else None)
    
    # =============================================================================
    # PER MODEL AND CLUSTER PLOTS 
    # - Model proportions across clusters, quality across clusters per model, 
    #   proportion delta across clusters per model
    # - Bar plots with models as hue
    # =============================================================================
    
    # Model proportions across clusters (same as per cluster proportions above, but categorized differently)
    plot_grouped_bar_by_hue(model_cluster_df, 'cluster', 'proportion', 'model',
                           'Model Proportions across Clusters', 
                           output_dir / 'per_model_cluster_proportions.png',
                           ci_lower_col=proportion_ci_lower, ci_upper_col=proportion_ci_upper,
                           wandb_key='per_model_cluster/proportions' if args.log_to_wandb else None)
    
    # Quality across clusters per model (for each quality metric)
    for metric in quality_metrics:
        quality_col = f'quality_{metric}'
        if quality_col in model_cluster_df.columns:
            ci_lower = f'{quality_col}_ci_lower' if f'{quality_col}_ci_lower' in model_cluster_df.columns else None
            ci_upper = f'{quality_col}_ci_upper' if f'{quality_col}_ci_upper' in model_cluster_df.columns else None
            
            plot_grouped_bar_by_hue(model_cluster_df, 'cluster', quality_col, 'model',
                                   f'Quality {metric.title()} across Clusters per Model', 
                                   output_dir / f'per_model_cluster_quality_{metric}.png',
                                   ci_lower_col=ci_lower, ci_upper_col=ci_upper,
                                   wandb_key=f'per_model_cluster/quality_{metric}' if args.log_to_wandb else None)
    
    # Proportion delta (salience) across clusters per model
    if 'proportion_delta' in model_cluster_df.columns:
        proportion_delta_ci_lower = 'proportion_delta_ci_lower' if 'proportion_delta_ci_lower' in model_cluster_df.columns else None
        proportion_delta_ci_upper = 'proportion_delta_ci_upper' if 'proportion_delta_ci_upper' in model_cluster_df.columns else None
        plot_grouped_bar_by_hue(model_cluster_df, 'cluster', 'proportion_delta', 'model',
                               'Proportion Delta (Salience) across Clusters per Model', 
                               output_dir / 'per_model_cluster_proportion_delta.png',
                               ci_lower_col=proportion_delta_ci_lower, ci_upper_col=proportion_delta_ci_upper,
                               wandb_key='per_model_cluster/proportion_delta' if args.log_to_wandb else None)
    
    # =============================================================================
    # ADDITIONAL PLOTS (original functionality - heatmaps and remaining plots)
    # =============================================================================
    
    # Heatmaps (these are unique visualizations not covered above)
    plot_heatmap(model_cluster_df, 'size', 'Conversation Count by Model-Cluster', 
                output_dir / 'model_cluster_size_heatmap.png', annot=True,
                wandb_key='heatmaps/model_cluster_size' if args.log_to_wandb else None)
    plot_heatmap(model_cluster_df, 'proportion', 'Proportion by Model-Cluster', 
                output_dir / 'model_cluster_proportion_heatmap.png',
                wandb_key='heatmaps/model_cluster_proportion' if args.log_to_wandb else None)
    
    if 'proportion_delta' in model_cluster_df.columns:
        significant_col = 'proportion_delta_significant' if 'proportion_delta_significant' in model_cluster_df.columns else None
        plot_heatmap(model_cluster_df, 'proportion_delta', 'Proportion Delta (Salience) by Model-Cluster', 
                    output_dir / 'model_cluster_proportion_delta_heatmap.png', significant_col=significant_col,
                    wandb_key='heatmaps/model_cluster_proportion_delta' if args.log_to_wandb else None)
    
    # Model-cluster quality plots (heatmaps only - these are unique visualizations)
    create_quality_plots(model_cluster_df, quality_metrics, output_dir, 'model_cluster', args.log_to_wandb)
    
    print(f"✅ All plots saved to {output_dir}")
    print(f"Generated {len(list(output_dir.glob('*.png')))} plot files")
    
    # Close wandb run if initialized
    if args.log_to_wandb and wandb.run:
        wandb.finish()
        print(f"✅ Wandb logging completed to project: {args.wandb_project}")

if __name__ == '__main__':
    main() 