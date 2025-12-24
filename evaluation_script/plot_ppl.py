#!/usr/bin/env python3
"""
Script to visualize PPL (Perplexity) trends from EAGLE inference log files.
Plots 10 separate subplots:
1. Target model cumulative PPL
2. Target model per-cycle PPL
3. Draft model cumulative PPL (if available)
4. Draft model per-cycle PPL (if available)
5. Average accept length per cycle
6. Total Surprisal = Accept_Len × log2(Draft_PPL)
7. Ratio = Draft_PPL / Target_PPL
8. Accept Length × Ratio
9. Endpoint Layer Score Sum (sum of scores at the accepted path's endpoint layer)
10. Accept Length × Endpoint Layer Score Sum

Also generates a second figure for stationarity analysis:
- Rolling statistics (mean and std)
- Coefficient of Variation (CV)
- ADF test results

Usage:
    python plot_ppl.py <log_file_path> [--output <output_image_path>]

Example:
    python plot_ppl.py eagle_cycle_log.jsonl --output ppl_plot.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Try to import statsmodels for ADF test
try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. ADF test will be skipped.")


def load_log_data(log_file: str) -> list:
    """Load and parse JSONL log file."""
    data = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
    return data


def extract_ppl_data(data: list) -> dict:
    """Extract PPL data from log entries."""
    cycles = []
    target_cumulative_ppl = []
    target_cycle_ppl = []
    draft_cumulative_ppl = []
    draft_cycle_ppl = []
    avg_accept_lengths = []
    endpoint_layer_score_sum = []  # New: sum of scores at the accepted path's endpoint layer
    
    # Track cumulative draft PPL (need to compute if not directly available)
    draft_total_logprob = 0.0
    draft_total_tokens = 0
    
    for entry in data:
        cycle = entry.get('cycle', len(cycles) + 1)
        cycles.append(cycle)
        
        # Target model PPL
        target_cumulative_ppl.append(entry.get('cumulative_ppl'))
        target_cycle_ppl.append(entry.get('current_cycle_ppl'))
        
        # Draft model PPL
        draft_cycle = entry.get('draft_current_cycle_ppl')
        draft_cycle_ppl.append(draft_cycle)
        
        # Accept length - use avg_accept_length directly
        avg_accept_length = entry.get('avg_accept_length')
        avg_accept_lengths.append(avg_accept_length)
        
        # Endpoint layer score sum
        endpoint_score = entry.get('avg_endpoint_layer_score_sum')
        endpoint_layer_score_sum.append(endpoint_score)
        
        # Compute cumulative draft PPL
        if draft_cycle is not None and draft_cycle > 0:
            accept_lengths = entry.get('accept_lengths', [])
            total_draft_tokens = sum(al for al in accept_lengths if al > 0)
            if total_draft_tokens > 0:
                cycle_mean_logprob = -np.log(draft_cycle)
                draft_total_logprob += cycle_mean_logprob * total_draft_tokens
                draft_total_tokens += total_draft_tokens
                cumulative_draft_ppl = np.exp(-draft_total_logprob / draft_total_tokens)
                draft_cumulative_ppl.append(cumulative_draft_ppl)
            else:
                draft_cumulative_ppl.append(None)
        else:
            draft_cumulative_ppl.append(None)
    
    # Compute Total Surprisal = Accept_Len × log2(Draft_PPL)
    total_surprisal = []
    for al, dppl in zip(avg_accept_lengths, draft_cycle_ppl):
        if al is not None and dppl is not None and not np.isnan(al) and not np.isnan(dppl) and dppl > 0:
            total_surprisal.append(al * np.log2(dppl))
        else:
            total_surprisal.append(None)
    
    # Compute Ratio = Draft_PPL / Target_PPL
    ppl_ratio = []
    for dppl, tppl in zip(draft_cycle_ppl, target_cycle_ppl):
        if dppl is not None and tppl is not None and not np.isnan(dppl) and not np.isnan(tppl) and tppl > 0:
            ppl_ratio.append(dppl / tppl)
        else:
            ppl_ratio.append(None)
    
    # Compute Accept Length × Ratio
    accept_length_times_ratio = []
    for al, ratio in zip(avg_accept_lengths, ppl_ratio):
        if al is not None and ratio is not None and not np.isnan(al) and not np.isnan(ratio):
            accept_length_times_ratio.append(al * ratio)
        else:
            accept_length_times_ratio.append(None)
    
    # Compute Accept Length × Endpoint Layer Score Sum
    accept_length_times_endpoint_score = []
    for al, score in zip(avg_accept_lengths, endpoint_layer_score_sum):
        if al is not None and score is not None and not np.isnan(al) and not np.isnan(score):
            accept_length_times_endpoint_score.append(al * score)
        else:
            accept_length_times_endpoint_score.append(None)
    
    return {
        'cycles': cycles,
        'target_cumulative_ppl': target_cumulative_ppl,
        'target_cycle_ppl': target_cycle_ppl,
        'draft_cumulative_ppl': draft_cumulative_ppl,
        'draft_cycle_ppl': draft_cycle_ppl,
        'avg_accept_lengths': avg_accept_lengths,
        'total_surprisal': total_surprisal,
        'ppl_ratio': ppl_ratio,
        'accept_length_times_ratio': accept_length_times_ratio,
        'endpoint_layer_score_sum': endpoint_layer_score_sum,
        'accept_length_times_endpoint_score': accept_length_times_endpoint_score,
    }


def filter_data(x_data, y_data):
    """Filter out None, NaN, Inf values."""
    filtered_x, filtered_y = [], []
    for x, y in zip(x_data, y_data):
        if y is not None and not np.isnan(y) and not np.isinf(y):
            filtered_x.append(x)
            filtered_y.append(y)
    return filtered_x, filtered_y


def filter_data_with_outlier_removal(x_data, y_data, iqr_multiplier=1.5):
    """Filter out None, NaN, Inf values and remove outliers using IQR method."""
    # First filter out None, NaN, Inf
    filtered_x, filtered_y = [], []
    for x, y in zip(x_data, y_data):
        if y is not None and not np.isnan(y) and not np.isinf(y):
            filtered_x.append(x)
            filtered_y.append(y)
    
    if len(filtered_y) < 4:
        return filtered_x, filtered_y
    
    # Calculate IQR
    q1 = np.percentile(filtered_y, 25)
    q3 = np.percentile(filtered_y, 75)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    # Filter outliers
    final_x, final_y = [], []
    for x, y in zip(filtered_x, filtered_y):
        if lower_bound <= y <= upper_bound:
            final_x.append(x)
            final_y.append(y)
    
    return final_x, final_y


def compute_rolling_statistics(data, window=100):
    """Compute rolling mean and rolling standard deviation."""
    data = np.array(data)
    n = len(data)
    if n < window:
        window = max(1, n // 10)
    
    rolling_mean = []
    rolling_std = []
    
    for i in range(n):
        start_idx = max(0, i - window + 1)
        window_data = data[start_idx:i+1]
        rolling_mean.append(np.mean(window_data))
        rolling_std.append(np.std(window_data))
    
    return np.array(rolling_mean), np.array(rolling_std)


def compute_coefficient_of_variation(data):
    """Compute Coefficient of Variation (CV) = std / mean."""
    data = np.array([d for d in data if d is not None and not np.isnan(d) and not np.isinf(d)])
    if len(data) == 0 or np.mean(data) == 0:
        return None
    return np.std(data) / np.mean(data)


def compute_adf_test(data):
    """Perform Augmented Dickey-Fuller test for stationarity."""
    if not HAS_STATSMODELS:
        return None
    
    data = np.array([d for d in data if d is not None and not np.isnan(d) and not np.isinf(d)])
    if len(data) < 20:
        return None
    
    try:
        result = adfuller(data, autolag='AIC')
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'used_lag': result[2],
            'nobs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    except Exception as e:
        print(f"ADF test failed: {e}")
        return None


def plot_stationarity_analysis(ppl_data: dict, output_path: str = None, window: int = 100):
    """Plot stationarity analysis for Accept Length and Total Surprisal."""
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle('Stationarity Analysis: Accept Length vs Total Surprisal', fontsize=16, fontweight='bold')
    
    cycles = ppl_data['cycles']
    
    # Get filtered data for both series
    x_accept, y_accept = filter_data(cycles, ppl_data['avg_accept_lengths'])
    x_surprisal, y_surprisal = filter_data(cycles, ppl_data['total_surprisal'])
    
    y_accept = np.array(y_accept)
    y_surprisal = np.array(y_surprisal)
    
    # Compute rolling statistics
    if len(y_accept) > 0:
        rolling_mean_accept, rolling_std_accept = compute_rolling_statistics(y_accept, window)
    if len(y_surprisal) > 0:
        rolling_mean_surprisal, rolling_std_surprisal = compute_rolling_statistics(y_surprisal, window)
    
    # Compute CV
    cv_accept = compute_coefficient_of_variation(y_accept)
    cv_surprisal = compute_coefficient_of_variation(y_surprisal)
    
    # Compute ADF test
    adf_accept = compute_adf_test(y_accept)
    adf_surprisal = compute_adf_test(y_surprisal)
    
    # ========== Row 1: Original data with rolling mean ==========
    # Left: Accept Length
    ax1 = axes[0, 0]
    if len(x_accept) > 0:
        ax1.plot(x_accept, y_accept, 'g-', linewidth=0.5, alpha=0.5, label='Original')
        ax1.plot(x_accept, rolling_mean_accept, 'b-', linewidth=2, label=f'Rolling Mean (w={window})')
        ax1.set_ylabel('Accept Length', fontsize=10)
        ax1.set_title(f'Accept Length with Rolling Mean\nCV = {cv_accept:.4f}' if cv_accept else 'Accept Length with Rolling Mean', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Cycle', fontsize=10)
    
    # Right: Total Surprisal
    ax2 = axes[0, 1]
    if len(x_surprisal) > 0:
        ax2.plot(x_surprisal, y_surprisal, 'm-', linewidth=0.5, alpha=0.5, label='Original')
        ax2.plot(x_surprisal, rolling_mean_surprisal, 'b-', linewidth=2, label=f'Rolling Mean (w={window})')
        ax2.set_ylabel('Total Surprisal (bits)', fontsize=10)
        ax2.set_title(f'Total Surprisal with Rolling Mean\nCV = {cv_surprisal:.4f}' if cv_surprisal else 'Total Surprisal with Rolling Mean', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Cycle', fontsize=10)
    
    # ========== Row 2: Rolling Mean comparison ==========
    # Left: Accept Length Rolling Mean
    ax3 = axes[1, 0]
    if len(x_accept) > 0:
        ax3.plot(x_accept, rolling_mean_accept, 'b-', linewidth=2)
        ax3.axhline(y=np.mean(y_accept), color='r', linestyle='--', linewidth=1.5, label=f'Global Mean = {np.mean(y_accept):.2f}')
        ax3.fill_between(x_accept, 
                         rolling_mean_accept - rolling_std_accept, 
                         rolling_mean_accept + rolling_std_accept, 
                         alpha=0.3, color='blue', label='±1 Rolling Std')
        ax3.set_ylabel('Rolling Mean', fontsize=10)
        ax3.set_title(f'Accept Length: Rolling Mean (window={window})', fontsize=12)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Cycle', fontsize=10)
    
    # Right: Total Surprisal Rolling Mean
    ax4 = axes[1, 1]
    if len(x_surprisal) > 0:
        ax4.plot(x_surprisal, rolling_mean_surprisal, 'b-', linewidth=2)
        ax4.axhline(y=np.mean(y_surprisal), color='r', linestyle='--', linewidth=1.5, label=f'Global Mean = {np.mean(y_surprisal):.2f}')
        ax4.fill_between(x_surprisal, 
                         rolling_mean_surprisal - rolling_std_surprisal, 
                         rolling_mean_surprisal + rolling_std_surprisal, 
                         alpha=0.3, color='blue', label='±1 Rolling Std')
        ax4.set_ylabel('Rolling Mean', fontsize=10)
        ax4.set_title(f'Total Surprisal: Rolling Mean (window={window})', fontsize=12)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('Cycle', fontsize=10)
    
    # ========== Row 3: Rolling Standard Deviation ==========
    # Left: Accept Length Rolling Std
    ax5 = axes[2, 0]
    if len(x_accept) > 0:
        ax5.plot(x_accept, rolling_std_accept, 'orange', linewidth=1.5)
        ax5.axhline(y=np.std(y_accept), color='r', linestyle='--', linewidth=1.5, label=f'Global Std = {np.std(y_accept):.2f}')
        ax5.set_ylabel('Rolling Std', fontsize=10)
        ax5.set_title(f'Accept Length: Rolling Std (window={window})', fontsize=12)
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlabel('Cycle', fontsize=10)
    
    # Right: Total Surprisal Rolling Std
    ax6 = axes[2, 1]
    if len(x_surprisal) > 0:
        ax6.plot(x_surprisal, rolling_std_surprisal, 'orange', linewidth=1.5)
        ax6.axhline(y=np.std(y_surprisal), color='r', linestyle='--', linewidth=1.5, label=f'Global Std = {np.std(y_surprisal):.2f}')
        ax6.set_ylabel('Rolling Std', fontsize=10)
        ax6.set_title(f'Total Surprisal: Rolling Std (window={window})', fontsize=12)
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlabel('Cycle', fontsize=10)
    
    # ========== Row 4: ADF Test Results and CV Comparison ==========
    # Left: ADF Test Summary Table
    ax7 = axes[3, 0]
    ax7.axis('off')
    
    # Create ADF test summary text
    adf_text = "Augmented Dickey-Fuller (ADF) Test Results\n"
    adf_text += "=" * 50 + "\n\n"
    adf_text += "Null Hypothesis (H0): The series has a unit root (non-stationary)\n"
    adf_text += "Alternative (H1): The series is stationary\n\n"
    
    if adf_accept:
        adf_text += "Accept Length:\n"
        adf_text += f"  ADF Statistic: {adf_accept['adf_statistic']:.4f}\n"
        adf_text += f"  p-value: {adf_accept['p_value']:.6f}\n"
        adf_text += f"  Critical Values:\n"
        for key, value in adf_accept['critical_values'].items():
            adf_text += f"    {key}: {value:.4f}\n"
        adf_text += f"  Result: {'STATIONARY ✓' if adf_accept['is_stationary'] else 'NON-STATIONARY ✗'}\n\n"
    else:
        adf_text += "Accept Length: ADF test not available\n\n"
    
    if adf_surprisal:
        adf_text += "Total Surprisal:\n"
        adf_text += f"  ADF Statistic: {adf_surprisal['adf_statistic']:.4f}\n"
        adf_text += f"  p-value: {adf_surprisal['p_value']:.6f}\n"
        adf_text += f"  Critical Values:\n"
        for key, value in adf_surprisal['critical_values'].items():
            adf_text += f"    {key}: {value:.4f}\n"
        adf_text += f"  Result: {'STATIONARY ✓' if adf_surprisal['is_stationary'] else 'NON-STATIONARY ✗'}\n"
    else:
        adf_text += "Total Surprisal: ADF test not available\n"
    
    ax7.text(0.05, 0.95, adf_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax7.set_title('ADF Test for Stationarity', fontsize=12)
    
    # Right: CV Comparison Bar Chart
    ax8 = axes[3, 1]
    cv_names = ['Accept Length', 'Total Surprisal']
    cv_values = [cv_accept if cv_accept else 0, cv_surprisal if cv_surprisal else 0]
    colors = ['green', 'magenta']
    
    bars = ax8.bar(cv_names, cv_values, color=colors, alpha=0.7, edgecolor='black')
    ax8.set_ylabel('Coefficient of Variation (CV)', fontsize=10)
    ax8.set_title('CV Comparison: Lower = More Stable\n(CV = σ / μ)', fontsize=12)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, cv_values):
        height = bar.get_height()
        ax8.annotate(f'{val:.4f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add interpretation text
    if cv_accept and cv_surprisal:
        if cv_surprisal < cv_accept:
            interpretation = f"Total Surprisal is {cv_accept/cv_surprisal:.1f}x more stable than Accept Length"
        else:
            interpretation = f"Accept Length is {cv_surprisal/cv_accept:.1f}x more stable than Total Surprisal"
        ax8.text(0.5, 0.02, interpretation, transform=ax8.transAxes, fontsize=10,
                 ha='center', style='italic', color='navy')
    
    plt.tight_layout()
    
    if output_path:
        # Generate second figure path
        path = Path(output_path)
        stationarity_path = str(path.parent / (path.stem + '_stationarity' + path.suffix))
        plt.savefig(stationarity_path, dpi=150, bbox_inches='tight')
        print(f"Stationarity analysis plot saved to: {stationarity_path}")
    
    plt.show()
    
    return {
        'cv_accept_length': cv_accept,
        'cv_total_surprisal': cv_surprisal,
        'adf_accept_length': adf_accept,
        'adf_total_surprisal': adf_surprisal
    }


def print_stationarity_summary(stats: dict):
    """Print stationarity analysis summary."""
    print("\n" + "="*60)
    print("Stationarity Analysis Summary")
    print("="*60)
    
    print("\n1. Coefficient of Variation (CV = σ/μ):")
    print("   (Lower CV = more stable relative to mean)")
    if stats['cv_accept_length']:
        print(f"   Accept Length CV:     {stats['cv_accept_length']:.4f}")
    if stats['cv_total_surprisal']:
        print(f"   Total Surprisal CV:   {stats['cv_total_surprisal']:.4f}")
    
    if stats['cv_accept_length'] and stats['cv_total_surprisal']:
        ratio = stats['cv_accept_length'] / stats['cv_total_surprisal']
        print(f"\n   → Total Surprisal is {ratio:.2f}x more stable than Accept Length")
    
    print("\n2. ADF Test (p < 0.05 indicates stationarity):")
    if stats['adf_accept_length']:
        adf = stats['adf_accept_length']
        status = "STATIONARY ✓" if adf['is_stationary'] else "NON-STATIONARY ✗"
        print(f"   Accept Length:    p-value = {adf['p_value']:.6f} → {status}")
    
    if stats['adf_total_surprisal']:
        adf = stats['adf_total_surprisal']
        status = "STATIONARY ✓" if adf['is_stationary'] else "NON-STATIONARY ✗"
        print(f"   Total Surprisal:  p-value = {adf['p_value']:.6f} → {status}")
    
    print("\n" + "="*60)


def plot_ppl(ppl_data: dict, output_path: str = None, title: str = "PPL Trends"):
    """Plot PPL trends with 10 separate subplots."""
    fig, axes = plt.subplots(5, 2, figsize=(16, 20))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    cycles = ppl_data['cycles']
    
    # Subplot 1: Target Model Cumulative PPL
    ax1 = axes[0, 0]
    x, y = filter_data(cycles, ppl_data['target_cumulative_ppl'])
    if x:
        ax1.plot(x, y, 'b-', linewidth=1.5, marker='o', markersize=2, alpha=0.8)
        ax1.set_ylabel('PPL', fontsize=10)
        ax1.set_title('Target Model Cumulative PPL', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Cycle', fontsize=10)
    else:
        ax1.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Target Model Cumulative PPL', fontsize=12)
    
    # Subplot 2: Target Model Per-Cycle PPL
    ax2 = axes[0, 1]
    x, y = filter_data(cycles, ppl_data['target_cycle_ppl'])
    if x:
        ax2.plot(x, y, 'b-', linewidth=1.5, marker='s', markersize=2, alpha=0.8)
        ax2.set_ylabel('PPL', fontsize=10)
        ax2.set_title('Target Model Per-Cycle PPL', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel('Cycle', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Target Model Per-Cycle PPL', fontsize=12)
    
    # Subplot 3: Draft Model Cumulative PPL
    ax3 = axes[1, 0]
    x, y = filter_data(cycles, ppl_data['draft_cumulative_ppl'])
    if x:
        ax3.plot(x, y, 'r-', linewidth=1.5, marker='o', markersize=2, alpha=0.8)
        ax3.set_ylabel('PPL', fontsize=10)
        ax3.set_title('Draft Model Cumulative PPL', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Cycle', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Draft Model Cumulative PPL', fontsize=12)
    
    # Subplot 4: Draft Model Per-Cycle PPL
    ax4 = axes[1, 1]
    x, y = filter_data(cycles, ppl_data['draft_cycle_ppl'])
    if x:
        ax4.plot(x, y, 'r-', linewidth=1.5, marker='s', markersize=2, alpha=0.8)
        ax4.set_ylabel('PPL', fontsize=10)
        ax4.set_title('Draft Model Per-Cycle PPL', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('Cycle', fontsize=10)
        ax4.set_ylim(0, 100)  # Cap at 100 for visibility
    else:
        ax4.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Draft Model Per-Cycle PPL', fontsize=12)
    
    # Subplot 5: Average Accept Length
    ax5 = axes[2, 0]
    x, y = filter_data(cycles, ppl_data['avg_accept_lengths'])
    if x:
        ax5.plot(x, y, 'g-', linewidth=1.5, marker='^', markersize=2, alpha=0.8)
        ax5.set_ylabel('Accept Length', fontsize=10)
        ax5.set_title('Average Accept Length per Cycle', fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.set_xlabel('Cycle', fontsize=10)
        # Set reasonable Y-axis limit for accept length (0 to max_spec_steps + 1)
        if y:
            max_val = max(y)
            ax5.set_ylim(0, max(max_val * 1.1, 1))
    else:
        ax5.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Average Accept Length per Cycle', fontsize=12)
    
    # Subplot 6: Total Surprisal = Accept_Len × log2(Draft_PPL)
    ax6 = axes[2, 1]
    x, y = filter_data(cycles, ppl_data['total_surprisal'])
    if x:
        ax6.plot(x, y, 'm-', linewidth=1.5, marker='d', markersize=2, alpha=0.8)
        ax6.set_ylabel('Total Surprisal (bits)', fontsize=10)
        ax6.set_title(r'Total Surprisal = Accept_Len × $\log_2$(Draft_PPL)', fontsize=12)
        ax6.grid(True, alpha=0.3)
        ax6.set_xlabel('Cycle', fontsize=10)
    else:
        ax6.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title(r'Total Surprisal = Accept_Len × $\log_2$(Draft_PPL)', fontsize=12)
    
    # Subplot 7: Ratio = Draft_PPL / Target_PPL
    ax7 = axes[3, 0]
    x, y = filter_data(cycles, ppl_data['ppl_ratio'])
    if x:
        ax7.plot(x, y, 'c-', linewidth=1.5, marker='p', markersize=2, alpha=0.8)
        ax7.set_ylabel('Ratio', fontsize=10)
        ax7.set_title('Ratio = Draft_PPL / Target_PPL', fontsize=12)
        ax7.grid(True, alpha=0.3)
        ax7.set_xlabel('Cycle', fontsize=10)
        ax7.set_ylim(0, 100)  # Cap at 100 for visibility
    else:
        ax7.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Ratio = Draft_PPL / Target_PPL', fontsize=12)
    
    # Subplot 8: Accept Length × Ratio (with outlier removal)
    ax8 = axes[3, 1]
    x, y = filter_data_with_outlier_removal(cycles, ppl_data['accept_length_times_ratio'])
    if x:
        ax8.plot(x, y, 'orange', linewidth=1.5, marker='*', markersize=2, alpha=0.8)
        ax8.set_ylabel('Accept Length × Ratio', fontsize=10)
        ax8.set_title('Accept Length × Ratio (outliers removed)', fontsize=12)
        ax8.grid(True, alpha=0.3)
        ax8.set_xlabel('Cycle', fontsize=10)
    else:
        ax8.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Accept Length × Ratio (outliers removed)', fontsize=12)
    
    # Subplot 9: Endpoint Layer Score Sum
    ax9 = axes[4, 0]
    x, y = filter_data(cycles, ppl_data['endpoint_layer_score_sum'])
    if x:
        ax9.plot(x, y, 'purple', linewidth=1.5, marker='h', markersize=2, alpha=0.8)
        ax9.set_ylabel('Score Sum', fontsize=10)
        ax9.set_title('Endpoint Layer Score Sum', fontsize=12)
        ax9.grid(True, alpha=0.3)
        ax9.set_xlabel('Cycle', fontsize=10)
    else:
        ax9.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Endpoint Layer Score Sum', fontsize=12)
    
    # Subplot 10: Accept Length × Endpoint Layer Score Sum
    ax10 = axes[4, 1]
    x, y = filter_data(cycles, ppl_data['accept_length_times_endpoint_score'])
    if x:
        ax10.plot(x, y, 'brown', linewidth=1.5, marker='v', markersize=2, alpha=0.8)
        ax10.set_ylabel('Accept Len × Score Sum', fontsize=10)
        ax10.set_title('Accept Length × Endpoint Layer Score Sum', fontsize=12)
        ax10.grid(True, alpha=0.3)
        ax10.set_xlabel('Cycle', fontsize=10)
    else:
        ax10.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Accept Length × Endpoint Layer Score Sum', fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()


def print_summary(ppl_data: dict):
    """Print summary statistics of PPL data."""
    print("\n" + "="*60)
    print("PPL Summary Statistics")
    print("="*60)
    
    def calc_stats(values, name):
        filtered = [v for v in values 
                    if v is not None and not np.isnan(v) and not np.isinf(v)]
        if filtered:
            print(f"\n{name}:")
            print(f"  Count:  {len(filtered)}")
            print(f"  Min:    {min(filtered):.4f}")
            print(f"  Max:    {max(filtered):.4f}")
            print(f"  Mean:   {np.mean(filtered):.4f}")
            print(f"  Final:  {filtered[-1]:.4f}")
        else:
            print(f"\n{name}: No data available")
    
    calc_stats(ppl_data['target_cumulative_ppl'], "Target Model Cumulative PPL")
    calc_stats(ppl_data['target_cycle_ppl'], "Target Model Per-Cycle PPL")
    calc_stats(ppl_data['draft_cumulative_ppl'], "Draft Model Cumulative PPL")
    calc_stats(ppl_data['draft_cycle_ppl'], "Draft Model Per-Cycle PPL")
    calc_stats(ppl_data['avg_accept_lengths'], "Average Accept Length")
    calc_stats(ppl_data['total_surprisal'], "Total Surprisal (Accept_Len × log2(Draft_PPL))")
    calc_stats(ppl_data['ppl_ratio'], "Ratio (Draft_PPL / Target_PPL)")
    calc_stats(ppl_data['accept_length_times_ratio'], "Accept Length × Ratio")
    calc_stats(ppl_data['endpoint_layer_score_sum'], "Endpoint Layer Score Sum")
    calc_stats(ppl_data['accept_length_times_endpoint_score'], "Accept Length × Endpoint Layer Score Sum")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PPL trends from EAGLE inference log files."
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="Path to the JSONL log file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output image path (e.g., ppl_plot.png). If not specified, only displays the plot."
    )
    parser.add_argument(
        "--title", "-t",
        type=str,
        default="EAGLE Inference PPL Trends",
        help="Title for the plot"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot (useful for headless environments)"
    )
    parser.add_argument(
        "--rolling-window", "-w",
        type=int,
        default=100,
        help="Window size for rolling statistics (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Check if log file exists
    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1
    
    print(f"Loading log file: {args.log_file}")
    data = load_log_data(args.log_file)
    print(f"Loaded {len(data)} log entries")
    
    if not data:
        print("Error: No valid log entries found")
        return 1
    
    ppl_data = extract_ppl_data(data)
    print_summary(ppl_data)
    
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    
    # Plot main PPL figure
    plot_ppl(ppl_data, output_path=args.output, title=args.title)
    
    # Plot stationarity analysis figure
    stationarity_stats = plot_stationarity_analysis(ppl_data, output_path=args.output, window=args.rolling_window)
    print_stationarity_summary(stationarity_stats)
    
    return 0


if __name__ == "__main__":
    exit(main())
