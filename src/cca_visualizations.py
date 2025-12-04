"""
CCA Visualization Utilities

This module provides visualization functions for Canonical Correlation Analysis results,
including correlation decay plots, demographic loading heatmaps, component rankings,
and method comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
from matplotlib.patches import Circle
from matplotlib_venn import venn2
import warnings

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


def plot_canonical_correlations(
    cca_results,
    output_dir: Path,
    demographic: str,
    run_id: str,
    fold_idx: int,
    n_dims: int = 10
):
    """
    Plot canonical correlation decay (scree plot).

    Shows how canonical correlations decrease across dimensions,
    helping identify the number of meaningful shared dimensions.

    Args:
        cca_results: CCAAnalysisResults object
        output_dir: Directory to save plot
        demographic: Demographic being analyzed
        run_id: Run identifier
        fold_idx: Fold index
        n_dims: Number of dimensions to plot (default: 10)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get canonical correlations from first component (they're all the same)
    if len(cca_results.component_results) == 0:
        warnings.warn("No CCA results to plot")
        return

    first_result = cca_results.component_results[0]
    canonical_corrs = first_result.canonical_correlations[:n_dims]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot as bar chart
    x = np.arange(1, len(canonical_corrs) + 1)
    bars = ax.bar(x, canonical_corrs, color='steelblue', alpha=0.7, edgecolor='black')

    # Add line plot
    ax.plot(x, canonical_corrs, 'o-', color='darkblue', linewidth=2, markersize=8)

    # Styling
    ax.set_xlabel('Canonical Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Canonical Correlation', fontsize=12, fontweight='bold')
    ax.set_title(f'CCA Canonical Correlation Decay\n{demographic.capitalize()} - Fold {fold_idx + 1}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for i, (xi, corr) in enumerate(zip(x, canonical_corrs)):
        ax.text(xi, corr + 0.02, f'{corr:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save
    output_path = output_dir / f"{demographic}_fold{fold_idx + 1}_{run_id}_canonical_correlations.png"
    plt.savefig(output_path)
    plt.close()

    print(f"  Saved canonical correlations plot to {output_path}")


def plot_demographic_loadings(
    cca_results,
    output_dir: Path,
    demographic: str,
    run_id: str,
    fold_idx: int,
    top_k: int = 5,
    n_dims: int = 5
):
    """
    Heatmap showing which demographics drive each canonical dimension.

    Visualizes the right weights (v vectors) showing which demographic
    combinations are most encoded in each canonical dimension.

    Args:
        cca_results: CCAAnalysisResults object
        output_dir: Directory to save plot
        demographic: Demographic being analyzed
        run_id: Run identifier
        fold_idx: Fold index
        top_k: Number of top components to average over (default: 5)
        n_dims: Number of canonical dimensions to show (default: 5)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(cca_results.component_results) == 0:
        warnings.warn("No CCA results to plot")
        return

    # Average right_weights across top-k components
    # Shape: (demographic_dim, n_canonical_dims)
    top_results = cca_results.component_results[:top_k]

    # Stack right_weights from all top components
    all_weights = []
    for result in top_results:
        weights = result.right_weights[:, :n_dims]  # Take first n_dims dimensions
        all_weights.append(weights)

    # Average across components
    avg_weights = np.mean(all_weights, axis=0)  # Shape: (demographic_dim, n_dims)

    # Create demographic feature names
    # Assuming one-hot encoding: gender_Man, gender_Woman, age_Young, age_Adult, etc.
    # We'll try to infer from the size
    n_features = avg_weights.shape[0]

    # Common demographics and their typical sizes
    feature_names = [f"Feature {i+1}" for i in range(n_features)]

    # If we have access to the actual demographic columns, we could do better
    # For now, use generic names

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, n_features * 0.3)))

    # Transpose for better visualization (demographics as rows, dimensions as columns)
    sns.heatmap(
        avg_weights,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Loading'},
        xticklabels=[f'Dim {i+1}' for i in range(n_dims)],
        yticklabels=feature_names,
        annot=True if n_features < 20 else False,  # Only annotate if not too many features
        fmt='.3f',
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(f'Demographic Loadings on Canonical Dimensions\n{demographic.capitalize()} - Fold {fold_idx + 1} (Top {top_k} components)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Canonical Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Demographic Features', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = output_dir / f"{demographic}_fold{fold_idx + 1}_{run_id}_demographic_loadings.png"
    plt.savefig(output_path)
    plt.close()

    print(f"  Saved demographic loadings heatmap to {output_path}")


def plot_component_rankings(
    cca_results,
    probe_type: str,
    output_dir: Path,
    demographic: str,
    run_id: str,
    fold_idx: int,
    top_k: int = 20
):
    """
    Bar chart of top components by CCA validation score.

    Args:
        cca_results: CCAAnalysisResults object
        probe_type: 'attention' or 'mlp'
        output_dir: Directory to save plot
        demographic: Demographic being analyzed
        run_id: Run identifier
        fold_idx: Fold index
        top_k: Number of top components to show
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(cca_results.component_results) == 0:
        warnings.warn("No CCA results to plot")
        return

    # Get top components
    top_results = cca_results.component_results[:top_k]

    # Extract data
    if probe_type == 'attention':
        labels = [f"L{r.component_id[0]}H{r.component_id[1]}" for r in top_results]
    else:  # mlp
        labels = [f"Layer {r.component_id[0]}" for r in top_results]

    val_scores = [r.val_score for r in top_results]
    train_scores = [r.train_score for r in top_results]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, top_k * 0.3)))

    # Horizontal bar chart
    y_pos = np.arange(len(labels))
    bars1 = ax.barh(y_pos - 0.2, val_scores, 0.4, label='Validation', color='steelblue', alpha=0.8)
    bars2 = ax.barh(y_pos + 0.2, train_scores, 0.4, label='Training', color='coral', alpha=0.8)

    # Add value labels
    for i, (val, train) in enumerate(zip(val_scores, train_scores)):
        ax.text(val + 0.01, i - 0.2, f'{val:.3f}', va='center', fontsize=8)
        ax.text(train + 0.01, i + 0.2, f'{train:.3f}', va='center', fontsize=8)

    # Styling
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Top component at top
    ax.set_xlabel('CCA Canonical Correlation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Component', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_k} Components by CCA Score\n{demographic.capitalize()} - Fold {fold_idx + 1}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Save
    output_path = output_dir / f"{demographic}_fold{fold_idx + 1}_{run_id}_component_rankings.png"
    plt.savefig(output_path)
    plt.close()

    print(f"  Saved component rankings plot to {output_path}")


def plot_method_comparison(
    probing_results,
    cca_results,
    comparison: dict,
    probe_type: str,
    output_dir: Path,
    demographic: str,
    run_id: str,
    fold_idx: int,
    top_k: int = 20
):
    """
    Comparison visualization showing overlap between CCA and probing methods.

    Creates a multi-panel figure with:
    1. Venn diagram showing component overlap
    2. Rank correlation scatter plot
    3. Score distribution comparison

    Args:
        probing_results: CircuitProbingResults or MLPLayerProbingResults
        cca_results: CCAAnalysisResults object
        comparison: Comparison dict from compare_cca_vs_probing
        probe_type: 'attention' or 'mlp'
        output_dir: Directory to save plot
        demographic: Demographic being analyzed
        run_id: Run identifier
        fold_idx: Fold index
        top_k: Number of top components considered
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 5))

    # 1. Venn diagram
    ax1 = fig.add_subplot(131)

    # Get counts
    cca_only = comparison['cca_unique']
    probing_only = comparison['probing_unique']
    overlap = comparison['overlap_count']

    # Create Venn diagram
    try:
        venn = venn2(
            subsets=(cca_only, probing_only, overlap),
            set_labels=('CCA', 'Probing'),
            ax=ax1,
            set_colors=('steelblue', 'coral'),
            alpha=0.6
        )

        # Add title
        ax1.set_title(f'Component Overlap\n({overlap}/{top_k} shared)',
                     fontsize=12, fontweight='bold')
    except:
        # Fallback if venn2 fails
        ax1.text(0.5, 0.5, f'Overlap: {overlap}/{top_k}\nCCA unique: {cca_only}\nProbing unique: {probing_only}',
                ha='center', va='center', fontsize=11)
        ax1.set_title('Component Overlap', fontsize=12, fontweight='bold')
        ax1.axis('off')

    # 2. Rank correlation scatter (for common components)
    ax2 = fig.add_subplot(132)

    common_components = comparison['common_components']
    if len(common_components) > 0:
        # Get ranks for common components
        cca_top = cca_results.top_k_components[:top_k]
        if hasattr(probing_results, 'top_k_heads'):
            probing_top = probing_results.top_k_heads[:top_k]
        else:
            probing_top = [(layer,) for layer in probing_results.top_k_layers[:top_k]]

        cca_ranks = []
        probing_ranks = []

        for comp in common_components:
            if isinstance(comp, list):
                comp = tuple(comp)
            cca_ranks.append(cca_top.index(comp))
            probing_ranks.append(probing_top.index(comp))

        # Scatter plot
        ax2.scatter(cca_ranks, probing_ranks, s=100, alpha=0.6, color='purple', edgecolors='black')

        # Diagonal line
        max_rank = max(max(cca_ranks), max(probing_ranks))
        ax2.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.3, label='Perfect agreement')

        # Styling
        ax2.set_xlabel('CCA Rank', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Probing Rank', fontsize=11, fontweight='bold')
        ax2.set_title(f'Rank Correlation: {comparison["rank_correlation"]:.3f}\n(p={comparison["rank_p_value"]:.4f})',
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No common components', ha='center', va='center', fontsize=11)
        ax2.set_title('Rank Correlation', fontsize=12, fontweight='bold')
        ax2.axis('off')

    # 3. Score distribution comparison
    ax3 = fig.add_subplot(133)

    # Get scores
    cca_scores = [abs(r.val_score) for r in cca_results.component_results[:top_k]]

    if hasattr(probing_results, 'head_results'):
        probing_scores = [abs(r.mcc_score) for r in probing_results.head_results[:top_k]]
    else:
        probing_scores = [abs(r.mcc_score) for r in probing_results.layer_results[:top_k]]

    # Box plots
    bp = ax3.boxplot(
        [cca_scores, probing_scores],
        labels=['CCA', 'Probing'],
        patch_artist=True,
        widths=0.6
    )

    # Color boxes
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.6)

    # Styling
    ax3.set_ylabel('Absolute Score', fontsize=11, fontweight='bold')
    ax3.set_title('Score Distributions\n(Top 20 components)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add mean lines
    ax3.axhline(np.mean(cca_scores), color='steelblue', linestyle='--', alpha=0.5, label=f'CCA mean: {np.mean(cca_scores):.3f}')
    ax3.axhline(np.mean(probing_scores), color='coral', linestyle='--', alpha=0.5, label=f'Probing mean: {np.mean(probing_scores):.3f}')
    ax3.legend(fontsize=9, loc='upper right')

    # Overall title
    fig.suptitle(f'CCA vs Probing Method Comparison\n{demographic.capitalize()} - Fold {fold_idx + 1}',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_path = output_dir / f"{demographic}_fold{fold_idx + 1}_{run_id}_method_comparison.png"
    plt.savefig(output_path)
    plt.close()

    print(f"  Saved method comparison plot to {output_path}")
