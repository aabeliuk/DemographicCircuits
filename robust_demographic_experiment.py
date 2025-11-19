"""
Robust Demographic Intervention Experiment with K-Fold Validation

This script runs comprehensive demographic intervention experiments with:
1. Separate extraction and intervention phases
2. Memory-optimized extraction: probes per-fold and saves only top N components
3. K-fold cross-validation with NO DATA LEAKAGE (top N selection uses train folds only)
4. Flexible configuration via command-line arguments
5. Comprehensive result tracking and reporting (including probing results per fold)
6. Unique run identifiers to prevent overwriting results

Usage:
    # Extract circuits first (probes PER FOLD and saves only top N components)
    # Use --top_k_heads to control how many components to keep (default: 10)
    # Use --n_folds to control k-fold splitting (default: 4)
    python robust_demographic_experiment.py --model meta-llama/Llama-3.2-1B \\
        --probe_type attention --phase extract --demographics gender age race \\
        --run_id my_experiment_1 --top_k_heads 10 --n_folds 5

    # Run k-fold validation on interventions using the same run_id and n_folds
    # Intervention phase loads fold-specific extractions (no data leakage!)
    python robust_demographic_experiment.py --model meta-llama/Llama-3.2-1B \\
        --probe_type attention --phase intervene --demographics gender age race \\
        --intervention_strength 100.0 --n_folds 5 \\
        --run_id my_experiment_1

    # Run both phases sequentially (run_id auto-generated if not specified)
    python robust_demographic_experiment.py --model meta-llama/Llama-3.2-1B \\
        --probe_type attention --phase both --demographics gender age \\
        --top_k_heads 15 --n_folds 4

Note: The extraction phase does K-FOLD PROBING to avoid data leakage:
      - Questions are split into K folds
      - For each fold, top N components are selected using TRAIN questions only
      - Each fold gets its own extraction file with fold-specific top N
      - Intervention phase loads the matching fold files automatically
      - This ensures component selection NEVER sees test questions!

Output files (per demographic, per fold):
      - Filtered activations: {demographic}_{probe_type}_{run_id}_fold{N}_extractions.pkl
      - Probing results: {demographic}_fold{N}_{probe_type}_probing_results.csv/json
      - Visualizations: {demographic}_fold{N}_{probe_type}_spearman_correlations.png
        * For attention: Bar plot of top 50 heads + heatmap by layer/head
        * For MLP: Bar plots of spearman correlations and accuracies by layer
"""

import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
from pathlib import Path
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from scipy.stats import kendalltau
import warnings
warnings.filterwarnings('ignore')

# Import from existing modules
import sys
from pathlib import Path
# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from probing_classifier import AttentionHeadProber
from baukit_activation_extraction import (
    extract_full_activations_baukit,
    extract_mlp_activations_baukit,
    BAUKIT_AVAILABLE
)
from anes_association_learning import ANESAssociationLearner, ANES_2024_VARIABLES
from intervention_engine import CircuitInterventionEngine, InterventionConfig
from mlp_intervention_engine import MLPInterventionEngine, MLPInterventionConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default values
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_PROBE_TYPE = "attention"
DEFAULT_RIDGE_ALPHA = 1.0
DEFAULT_N_FOLDS = 4
DEFAULT_TOP_K_HEADS = 10
DEFAULT_N_SAMPLES_PER_CATEGORY = 100
DEFAULT_INTERVENTION_STRENGTH = 10.0
DEFAULT_EVAL_SAMPLE_SIZE = 50

# Available demographics and political questions
ALL_DEMOGRAPHICS = [
    'gender', 'age', 'race', 'education', 'marital_status',
    'income', 'ideology', 'religion', 'urban_rural'
]

ALL_POLITICAL_QUESTIONS = [
    'abortion', 'death_penalty', 'military_force', 'defense_spending',
    'govt_jobs', 'govt_help_blacks', 'colleges_opinion', 'dei_opinion',
    'journalist_access', 'transgender_bathrooms', 'birthright_citizenship',
    'immigration_policy'
]


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Robust Demographic Intervention Experiment with K-Fold Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model configuration
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                       help=f'HuggingFace model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu). Auto-detected if not specified.')

    # Experiment phase
    parser.add_argument('--phase', type=str, choices=['extract', 'intervene', 'both'],
                       default='both', help='Experiment phase to run (default: both)')

    # Probe configuration
    parser.add_argument('--probe_type', type=str, choices=['attention', 'mlp', 'both'],
                       default=DEFAULT_PROBE_TYPE,
                       help=f'Type of probe to use (default: {DEFAULT_PROBE_TYPE})')
    parser.add_argument('--ridge_alpha', type=float, default=DEFAULT_RIDGE_ALPHA,
                       help=f'Ridge regression alpha (default: {DEFAULT_RIDGE_ALPHA})')

    # K-fold validation
    parser.add_argument('--n_folds', type=int, default=DEFAULT_N_FOLDS,
                       help=f'Number of folds for cross-validation (default: {DEFAULT_N_FOLDS})')

    # Intervention configuration
    parser.add_argument('--top_k_heads', type=int, default=DEFAULT_TOP_K_HEADS,
                       help=f'Number of top heads/layers to use (default: {DEFAULT_TOP_K_HEADS})')
    parser.add_argument('--intervention_strength', type=float, default=DEFAULT_INTERVENTION_STRENGTH,
                       help=f'Intervention strength (default: {DEFAULT_INTERVENTION_STRENGTH})')
    parser.add_argument('--eval_sample_size', type=int, default=DEFAULT_EVAL_SAMPLE_SIZE,
                       help=f'Sample size per test question (default: {DEFAULT_EVAL_SAMPLE_SIZE})')

    # Data configuration
    parser.add_argument('--n_samples_per_category', type=int, default=DEFAULT_N_SAMPLES_PER_CATEGORY,
                       help=f'Samples per category for extraction (default: {DEFAULT_N_SAMPLES_PER_CATEGORY})')
    parser.add_argument('--anes_data_path', type=str,
                       default='anes_timeseries_2024_csv_20250808/anes_timeseries_2024_csv_20250808.csv',
                       help='Path to ANES data CSV')

    # Demographics to test
    parser.add_argument('--demographics', type=str, nargs='+', default=None,
                       help=f'Demographics to test (default: all). Options: {", ".join(ALL_DEMOGRAPHICS)}')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='robust_experiment_results',
                       help='Directory for output files (default: robust_experiment_results)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip demographics with existing extraction results')
    parser.add_argument('--run_id', type=str, default=None,
                       help='Run identifier for naming files. If not specified, uses timestamp. '
                            'For intervention phase, use the run_id from extraction phase.')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'

    # Use all demographics if not specified
    if args.demographics is None:
        args.demographics = ALL_DEMOGRAPHICS

    # Validate demographics
    invalid_demos = set(args.demographics) - set(ALL_DEMOGRAPHICS)
    if invalid_demos:
        parser.error(f"Invalid demographics: {invalid_demos}. Valid options: {ALL_DEMOGRAPHICS}")

    # Generate run_id if not specified
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    return args


# ============================================================================
# DATA LOADING
# ============================================================================

def load_anes_data(path: str) -> pd.DataFrame:
    """Load and preprocess ANES survey data"""
    anes_learner = ANESAssociationLearner(path)
    data = anes_learner.anes_data

    # Filter for binary gender if needed
    if 'gender' in data.columns:
        data = data[data['gender'].isin(['Man', 'Woman'])].copy()

    print(f"Loaded {len(data)} respondents")
    print(f"Available columns: {list(data.columns)[:10]}...")

    return data


# ============================================================================
# PROMPT GENERATION
# ============================================================================

def create_prompt_without_attribute(
    user_profile: pd.Series,
    question: str,
    exclude_attribute: str,
    answer_options: Optional[List[str]] = None,
    answer: Optional[str] = None
) -> str:
    """Build a demographic prompt WITHOUT a specified attribute"""
    adjectives = []
    gender = None
    education_phrase = None
    location_phrase = None

    # Mappings
    age_map = {'Young Adult': 'young', 'Adult': 'middle-aged', 'Senior': 'senior'}
    marital_map = {'Married': 'married', 'Previously married': 'previously married', 'Never married': 'never married'}
    religion_map = {'Religious': 'religious', 'Not Religious': 'non-religious'}
    income_map = {'Low': 'low-income', 'Middle': 'middle-income', 'High': 'high-income'}
    ideology_map = {'Left': 'politically liberal', 'Center': 'politically moderate', 'Right': 'politically conservative'}
    edu_map = {'Low': 'with a high school education', 'Medium': 'with some college', 'High': 'with a college degree'}

    # Build demographics, excluding the target attribute
    if exclude_attribute != 'age' and 'age' in user_profile and not pd.isna(user_profile['age']):
        adjectives.append(age_map.get(user_profile['age'], str(user_profile['age'])))

    if exclude_attribute != 'race' and 'race' in user_profile and not pd.isna(user_profile['race']):
        adjectives.append(str(user_profile['race']))

    if exclude_attribute != 'marital_status' and 'marital_status' in user_profile and not pd.isna(user_profile['marital_status']):
        adjectives.append(marital_map.get(user_profile['marital_status'], ''))

    if exclude_attribute != 'religion' and 'religion' in user_profile and not pd.isna(user_profile['religion']):
        adjectives.append(religion_map.get(user_profile['religion'], ''))

    if exclude_attribute != 'income' and 'income' in user_profile and not pd.isna(user_profile['income']):
        adjectives.append(income_map.get(user_profile['income'], ''))

    if exclude_attribute != 'ideology' and 'ideology' in user_profile and not pd.isna(user_profile['ideology']):
        adjectives.append(ideology_map.get(user_profile['ideology'], ''))

    if exclude_attribute != 'gender' and 'gender' in user_profile and not pd.isna(user_profile['gender']):
        gender = str(user_profile['gender']).lower()

    if exclude_attribute != 'education' and 'education' in user_profile and not pd.isna(user_profile['education']):
        education_phrase = edu_map.get(user_profile['education'], '')

    if exclude_attribute != 'urban_rural' and 'urban_rural' in user_profile and not pd.isna(user_profile['urban_rural']):
        location_phrase = f"from a {str(user_profile['urban_rural']).lower()} area"

    # Build final demographic string
    demographic_parts = []

    if adjectives:
        adjectives_str = ', '.join([a for a in adjectives if a])
        if gender:
            demographic_parts.append(f"{adjectives_str} {gender}")
        else:
            demographic_parts.append(f"{adjectives_str} person")
    elif gender:
        demographic_parts.append(gender)
    else:
        demographic_parts.append("person")

    if education_phrase:
        demographic_parts.append(education_phrase)

    if location_phrase:
        demographic_parts.append(location_phrase)

    demographic = ' '.join(demographic_parts)

    # Get question label
    question_label = ANES_2024_VARIABLES.get(question, {}).get('label', question)

    # Build prompt
    if answer is not None:
        if answer_options:
            options_str = ' / '.join(answer_options)
            prompt = f"A {demographic} is asked: {question_label} ({options_str}). They answer: {answer}"
        else:
            prompt = f"A {demographic} is asked: {question_label}. They answer: {answer}"
    else:
        if answer_options:
            options_str = ' / '.join(answer_options)
            prompt = f"A {demographic} is asked: {question_label} ({options_str}). They answer:"
        else:
            prompt = f"A {demographic} is asked: {question_label}. They answer:"

    return prompt


def create_prompt(
    user_profile: pd.Series,
    question: str,
    answer_options: Optional[List[str]] = None,
    answer: Optional[str] = None
) -> str:
    """Build a demographic prompt WITH all attributes"""
    return create_prompt_without_attribute(
        user_profile, question, None, answer_options, answer
    )


# ============================================================================
# PHASE 1: EXTRACTION
# ============================================================================

def extract_activations_for_question(
    model,
    tokenizer,
    df: pd.DataFrame,
    question: str,
    demographic_attr: str,
    n_samples_per_category: int,
    device: str,
    probe_type: str
) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray], List[str]]:
    """Extract activations for a single question"""

    # Filter for valid responses
    valid_df = df[df[question].notna()].copy()

    if demographic_attr not in valid_df.columns:
        print(f"  WARNING: Demographic '{demographic_attr}' not found")
        return None, None, []

    # Get unique categories
    category_names = sorted(valid_df[demographic_attr].dropna().unique().tolist())
    n_categories = len(category_names)

    if n_categories < 2:
        print(f"  WARNING: Insufficient categories ({n_categories})")
        return None, None, []

    # Sample from each category
    category_samples = []
    for category in category_names:
        category_df = valid_df[valid_df[demographic_attr] == category].copy()
        n_available = len(category_df)
        n_sample = min(n_samples_per_category, n_available)

        if n_sample < 10:
            print(f"  WARNING: Only {n_sample} samples for '{category}'")

        sampled_df = category_df.sample(n=n_sample, random_state=42)
        category_samples.append(sampled_df)

    # Use minimum sample size for balanced dataset
    min_samples = min(len(df_cat) for df_cat in category_samples)
    category_samples = [df_cat.sample(n=min_samples, random_state=42) for df_cat in category_samples]

    # Create prompts WITHOUT the target demographic
    all_prompts = []
    all_category_labels = []

    # Get answer options
    answer_options = None
    if question in ANES_2024_VARIABLES and 'values' in ANES_2024_VARIABLES[question]:
        answer_options = list(ANES_2024_VARIABLES[question]['values'].values())

    for category_idx, (category, sampled_df) in enumerate(zip(category_names, category_samples)):
        for idx, user_profile in sampled_df.iterrows():
            answer = user_profile[question]
            prompt = create_prompt_without_attribute(
                user_profile, question, demographic_attr, answer_options, answer
            )
            all_prompts.append(prompt)
            all_category_labels.append(category_idx)

    # Extract activations
    if not BAUKIT_AVAILABLE:
        raise ImportError("This experiment requires baukit. Install with: pip install baukit")

    print(f"  Extracting {probe_type} activations for {len(all_prompts)} samples...")

    if probe_type == 'attention':
        all_activations = extract_full_activations_baukit(
            model, tokenizer, all_prompts, device, aggregation='mean'
        )
    elif probe_type == 'mlp':
        all_activations = extract_mlp_activations_baukit(
            model, tokenizer, all_prompts, device, aggregation='mean'
        )
    elif probe_type == 'both':
        attn_activations = extract_full_activations_baukit(
            model, tokenizer, all_prompts, device, aggregation='mean'
        )
        mlp_activations = extract_mlp_activations_baukit(
            model, tokenizer, all_prompts, device, aggregation='mean'
        )
        all_activations = (attn_activations, mlp_activations)
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    category_labels = np.array(all_category_labels)

    return all_activations, category_labels, category_names


def probe_and_select_top_components(
    activations,
    labels,
    model_config,
    probe_type: str,
    ridge_alpha: float,
    top_k: int
) -> Dict:
    """
    Probe activations to identify top N layers/heads and extract intervention weights.

    Returns:
        Dictionary containing:
        - top_indices: List of (layer, head) tuples for attention or list of layers for MLP
        - probing_results: Full probing results
        - intervention_weights: Pre-computed intervention weights
    """
    print(f"\n  Running probing to identify top {top_k} components...")

    if probe_type == 'attention':
        prober = AttentionHeadProber(
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            head_dim=model_config['head_dim'],
            alpha=ridge_alpha,
            n_folds=2,
            task_type='classification',
            random_state=42
        )

        probing_results = prober.probe_all_heads(activations, labels, aggregation='mean')
        intervention_weights = prober.get_intervention_weights(probing_results, top_k=top_k)

        # Extract top head indices (both probing_results and head items are dataclasses)
        top_heads = probing_results.head_results[:top_k]
        top_indices = [(head.layer, head.head) for head in top_heads]

    elif probe_type == 'mlp':
        probing_results = probe_mlp_layers(
            activations, labels, model_config['num_layers'], ridge_alpha
        )
        intervention_weights = extract_mlp_intervention_weights(probing_results, top_k=top_k)

        # Extract top layer indices
        top_layers = probing_results['layer_results'][:top_k]
        top_indices = [layer['layer'] for layer in top_layers]

    else:
        raise ValueError(f"Unsupported probe_type: {probe_type}")

    print(f"  Selected top {len(top_indices)} components")

    return {
        'top_indices': top_indices,
        'probing_results': probing_results,
        'intervention_weights': intervention_weights
    }


def save_probing_results(
    probing_results,
    probe_type: str,
    output_path: Path,
    demographic: str
):
    """
    Save probing results to CSV and JSON formats.

    Args:
        probing_results: Results from probe_and_select_top_components (CircuitProbingResults or Dict)
        probe_type: 'attention' or 'mlp'
        output_path: Directory to save results
        demographic: Name of demographic being probed
    """
    output_path.mkdir(parents=True, exist_ok=True)

    if probe_type == 'attention':
        # Extract head results (from CircuitProbingResults dataclass)
        head_results = probing_results.head_results if hasattr(probing_results, 'head_results') else probing_results['head_results']

        # Create DataFrame
        results_list = []
        for head_info in head_results:
            # head_info is a ProbingResult dataclass, use attribute access
            if hasattr(head_info, 'layer'):
                results_list.append({
                    'layer': head_info.layer,
                    'head': head_info.head,
                    'accuracy': head_info.val_score,  # val_score is the accuracy
                    'spearman_r': head_info.spearman_r,
                    'p_value': head_info.spearman_p
                })
            else:
                # Fallback for dict format
                results_list.append({
                    'layer': head_info['layer'],
                    'head': head_info['head'],
                    'accuracy': head_info['accuracy'],
                    'spearman_r': head_info['spearman_r'],
                    'p_value': head_info['p_value']
                })

        df = pd.DataFrame(results_list)

        # Save CSV
        csv_path = output_path / f"{demographic}_attention_probing_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved probing results to: {csv_path}")

        # Save JSON with full details
        json_path = output_path / f"{demographic}_attention_probing_results.json"
        json_results = {
            'demographic': demographic,
            'probe_type': probe_type,
            'head_results': results_list,
            'timestamp': datetime.now().isoformat()
        }
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

    elif probe_type == 'mlp':
        # Extract layer results
        layer_results = probing_results['layer_results']

        # Create DataFrame
        results_list = []
        for layer_info in layer_results:
            results_list.append({
                'layer': layer_info['layer'],
                'accuracy': layer_info['accuracy'],
                'spearman_r': layer_info['spearman_r'],
                'p_value': layer_info['p_value']
            })

        df = pd.DataFrame(results_list)

        # Save CSV
        csv_path = output_path / f"{demographic}_mlp_probing_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved probing results to: {csv_path}")

        # Save JSON with full details
        json_path = output_path / f"{demographic}_mlp_probing_results.json"
        json_results = {
            'demographic': demographic,
            'probe_type': probe_type,
            'layer_results': results_list,
            'timestamp': datetime.now().isoformat()
        }
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

    return df


def plot_spearman_correlations(
    probing_results,
    probe_type: str,
    output_path: Path,
    demographic: str,
    top_k: int = None
):
    """
    Create and save visualization of spearman correlations.

    Args:
        probing_results: Results from probe_and_select_top_components (CircuitProbingResults or Dict)
        probe_type: 'attention' or 'mlp'
        output_path: Directory to save figure
        demographic: Name of demographic being probed
        top_k: Number of top components to highlight
    """
    output_path.mkdir(parents=True, exist_ok=True)

    if probe_type == 'attention':
        head_results = probing_results.head_results if hasattr(probing_results, 'head_results') else probing_results['head_results']

        # Extract data (handle both ProbingResult dataclass and dict)
        if head_results and hasattr(head_results[0], 'layer'):
            # ProbingResult dataclass
            layers = [h.layer for h in head_results]
            heads = [h.head for h in head_results]
            spearman_rs = [h.spearman_r for h in head_results]
        else:
            # Dict format
            layers = [h['layer'] for h in head_results]
            heads = [h['head'] for h in head_results]
            spearman_rs = [h['spearman_r'] for h in head_results]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Bar plot of all heads (sorted by spearman)
        if head_results and hasattr(head_results[0], 'layer'):
            x_labels = [f"L{h.layer}-H{h.head}" for h in head_results[:50]]  # Top 50
        else:
            x_labels = [f"L{h['layer']}-H{h['head']}" for h in head_results[:50]]  # Top 50
        if top_k:
            colors = ['red' if i < top_k else 'blue' for i in range(min(50, len(spearman_rs)))]
        else:
            colors = ['blue'] * min(50, len(spearman_rs))

        ax1.bar(range(len(x_labels)), spearman_rs[:50], color=colors, alpha=0.7)
        ax1.set_xlabel('Layer-Head', fontsize=12)
        ax1.set_ylabel('Spearman Correlation', fontsize=12)
        ax1.set_title(f'Top 50 Attention Heads by Spearman Correlation\n{demographic.title()}', fontsize=14)
        ax1.tick_params(axis='x', rotation=90, labelsize=8)
        ax1.set_xticks(range(len(x_labels)))
        ax1.set_xticklabels(x_labels)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Add legend
        if top_k:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label=f'Top {top_k} (used)'),
                Patch(facecolor='blue', alpha=0.7, label='Other heads')
            ]
            ax1.legend(handles=legend_elements, loc='upper right')

        # Plot 2: Heatmap of spearman correlations by layer and head
        num_layers = max(layers) + 1
        num_heads = max(heads) + 1

        # Create matrix (handle both ProbingResult dataclass and dict)
        spearman_matrix = np.zeros((num_layers, num_heads))
        for h in head_results:
            if hasattr(h, 'layer'):
                spearman_matrix[h.layer, h.head] = h.spearman_r
            else:
                spearman_matrix[h['layer'], h['head']] = h['spearman_r']

        im = ax2.imshow(spearman_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_xlabel('Head', fontsize=12)
        ax2.set_ylabel('Layer', fontsize=12)
        ax2.set_title(f'Spearman Correlation Heatmap\n{demographic.title()}', fontsize=14)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Spearman Correlation', fontsize=10)

        # Mark top K heads
        if top_k:
            for i, h in enumerate(head_results[:top_k]):
                if hasattr(h, 'layer'):
                    ax2.plot(h.head, h.layer, 'g*', markersize=10, markeredgecolor='yellow', markeredgewidth=1)
                else:
                    ax2.plot(h['head'], h['layer'], 'g*', markersize=10, markeredgecolor='yellow', markeredgewidth=1)

        plt.tight_layout()

        # Save figure
        fig_path = output_path / f"{demographic}_attention_spearman_correlations.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved correlation plot to: {fig_path}")

    elif probe_type == 'mlp':
        layer_results = probing_results['layer_results']

        # Extract data
        layers = [l['layer'] for l in layer_results]
        spearman_rs = [l['spearman_r'] for l in layer_results]
        accuracies = [l['accuracy'] for l in layer_results]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Spearman correlation by layer
        if top_k:
            colors = ['red' if i < top_k else 'blue' for i in range(len(layers))]
        else:
            colors = ['blue'] * len(layers)
        ax1.bar(layers, spearman_rs, color=colors, alpha=0.7)
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Spearman Correlation', fontsize=12)
        ax1.set_title(f'MLP Layer Spearman Correlations\n{demographic.title()}', fontsize=14)
        ax1.grid(axis='y', alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

        # Add legend
        if top_k:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label=f'Top {top_k} (used)'),
                Patch(facecolor='blue', alpha=0.7, label='Other layers')
            ]
            ax1.legend(handles=legend_elements, loc='upper right')

        # Plot 2: Accuracy by layer
        ax2.bar(layers, accuracies, color=colors, alpha=0.7)
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title(f'MLP Layer Classification Accuracy\n{demographic.title()}', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save figure
        fig_path = output_path / f"{demographic}_mlp_spearman_correlations.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved correlation plot to: {fig_path}")


def filter_activations_by_top_components(
    activations: torch.Tensor,
    top_indices: List,
    probe_type: str,
    model_config: Dict
) -> torch.Tensor:
    """
    Filter activations to keep only top N layers/heads.

    For attention: activations shape [batch, num_layers, num_heads, head_dim]
    For MLP: activations shape [batch, num_layers, hidden_dim]

    Returns filtered activations with reduced dimensions.
    """
    if probe_type == 'attention':
        # top_indices is list of (layer, head) tuples
        # Extract only those specific heads
        batch_size = activations.shape[0]
        head_dim = activations.shape[3]

        filtered_list = []
        for layer, head in top_indices:
            # Extract activations for this specific head: [batch, head_dim]
            head_activations = activations[:, layer, head, :]
            filtered_list.append(head_activations)

        # Stack: [batch, top_k, head_dim]
        filtered_activations = torch.stack(filtered_list, dim=1)

    elif probe_type == 'mlp':
        # top_indices is list of layer indices
        # Extract only those specific layers
        filtered_list = []
        for layer in top_indices:
            layer_activations = activations[:, layer, :]
            filtered_list.append(layer_activations)

        # Stack: [batch, top_k, hidden_dim]
        filtered_activations = torch.stack(filtered_list, dim=1)

    else:
        raise ValueError(f"Unsupported probe_type: {probe_type}")

    return filtered_activations


def run_extraction_phase(args):
    """
    Phase 1: Extract activations for all questions and save to disk.

    Now includes probing to identify top N layers/heads and only saves
    activations for those components to reduce memory usage.

    This allows interventions to be tested with different configurations
    without re-extracting activations.
    """
    print("\n" + "="*80)
    print("PHASE 1: EXTRACTION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Probe type: {args.probe_type}")
    print(f"Demographics: {args.demographics}")
    print(f"Device: {args.device}")
    print(f"Run ID: {args.run_id}")
    print("="*80 + "\n")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir) / 'extractions'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading ANES data...")
    df = load_anes_data(args.anes_data_path)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on device: {args.device}")

    # Get model config for probing
    model_config = {
        'num_layers': model.config.num_hidden_layers,
        'num_heads': model.config.num_attention_heads,
        'head_dim': model.config.hidden_size // model.config.num_attention_heads,
        'hidden_size': model.config.hidden_size
    }

    # Process each demographic
    for demographic in args.demographics:
        print(f"\n{'='*80}")
        print(f"EXTRACTING: {demographic.upper()}")
        print(f"{'='*80}")

        # Check if already extracted
        extraction_file = output_dir / f"{demographic}_{args.probe_type}_{args.run_id}_extractions.pkl"
        if args.skip_existing and extraction_file.exists():
            print(f"Skipping - extraction already exists: {extraction_file}")
            continue

        # Extract activations for all questions
        question_extractions = {}

        for q_idx, question in enumerate(ALL_POLITICAL_QUESTIONS):
            q_label = ANES_2024_VARIABLES.get(question, {}).get('label', question)
            print(f"\nQuestion {q_idx + 1}/{len(ALL_POLITICAL_QUESTIONS)}: {question}")
            print(f"  {q_label}")

            try:
                activations, labels, category_names = extract_activations_for_question(
                    model, tokenizer, df, question, demographic,
                    args.n_samples_per_category, args.device, args.probe_type
                )

                if activations is None:
                    print(f"  Skipping due to insufficient data")
                    continue

                question_extractions[question] = {
                    'activations': activations,
                    'labels': labels,
                    'category_names': category_names
                }

                print(f"  Extracted: {activations.shape if not isinstance(activations, tuple) else (activations[0].shape, activations[1].shape)}")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        # Check if we have any questions extracted
        if len(question_extractions) == 0:
            print(f"\nNo questions extracted for {demographic}, skipping...")
            continue

        # Skip probing for 'both' mode (not supported for intervention)
        if args.probe_type == 'both':
            print(f"\nWARNING: probe_type='both' does not support probing/filtering. Saving full activations.")
            # Save without probing (old behavior)
            extraction_data = {
                'demographic': demographic,
                'probe_type': args.probe_type,
                'model': args.model,
                'run_id': args.run_id,
                'n_samples_per_category': args.n_samples_per_category,
                'question_extractions': question_extractions,
                'timestamp': datetime.now().isoformat()
            }
            with open(extraction_file, 'wb') as f:
                pickle.dump(extraction_data, f)
            print(f"\nSaved extractions for {len(question_extractions)} questions: {extraction_file}")
        else:
            # K-FOLD PROBING: Select top N components per fold to avoid data leakage
            print(f"\n{'='*60}")
            print(f"K-FOLD PROBING FOR {demographic.upper()}")
            print(f"{'='*60}")

            questions = list(question_extractions.keys())
            first_question = questions[0]
            category_names = question_extractions[first_question]['category_names']

            # Setup k-fold cross-validation
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

            print(f"\nSplitting {len(questions)} questions into {args.n_folds} folds")
            print(f"Each fold will select top {args.top_k_heads} components using train questions only\n")

            # Process each fold
            for fold_idx, (train_indices, test_indices) in enumerate(kfold.split(questions)):
                print(f"\n{'-'*80}")
                print(f"FOLD {fold_idx + 1}/{args.n_folds}")
                print(f"{'-'*80}")

                train_questions = [questions[i] for i in train_indices]
                test_questions = [questions[i] for i in test_indices]

                print(f"Train questions ({len(train_questions)}): {train_questions}")
                print(f"Test questions ({len(test_questions)}): {test_questions}")

                # Concatenate activations from TRAIN questions only
                train_activations_list = []
                train_labels_list = []

                for question in train_questions:
                    q_data = question_extractions[question]
                    train_activations_list.append(q_data['activations'])
                    train_labels_list.append(q_data['labels'])

                # Concatenate
                concatenated_activations = torch.cat(train_activations_list, dim=0)
                concatenated_labels = np.concatenate(train_labels_list)

                print(f"\nTrain activations shape: {concatenated_activations.shape}")
                print(f"Train samples: {len(concatenated_labels)}")

                # Run probing on TRAIN questions only to select top N
                probing_data = probe_and_select_top_components(
                    concatenated_activations,
                    concatenated_labels,
                    model_config,
                    args.probe_type,
                    args.ridge_alpha,
                    args.top_k_heads
                )

                top_indices = probing_data['top_indices']

                # Save probing results and visualizations for this fold
                print(f"\nSaving fold {fold_idx + 1} probing results and visualizations...")
                results_output_dir = output_dir / 'probing_results'

                # Save results to CSV/JSON
                save_probing_results(
                    probing_data['probing_results'],
                    args.probe_type,
                    results_output_dir,
                    f"{demographic}_fold{fold_idx + 1}"
                )

                # Create and save visualization
                plot_spearman_correlations(
                    probing_data['probing_results'],
                    args.probe_type,
                    results_output_dir,
                    f"{demographic}_fold{fold_idx + 1}",
                    args.top_k_heads
                )

                # Clean up concatenated data to free memory
                del concatenated_activations
                del concatenated_labels
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                # Filter ALL questions' activations to keep only this fold's top N
                print(f"\nFiltering all questions to fold {fold_idx + 1}'s top {args.top_k_heads} components...")
                fold_question_extractions = {}
                for question, q_data in question_extractions.items():
                    original_shape = q_data['activations'].shape
                    filtered_activations = filter_activations_by_top_components(
                        q_data['activations'],
                        top_indices,
                        args.probe_type,
                        model_config
                    )
                    fold_question_extractions[question] = {
                        'activations': filtered_activations,
                        'labels': q_data['labels'],
                        'category_names': q_data['category_names']
                    }
                    if question in train_questions[:3]:  # Show first 3 train
                        print(f"  [Train] {question}: {original_shape} -> {filtered_activations.shape}")
                    elif question in test_questions[:2]:  # Show first 2 test
                        print(f"  [Test]  {question}: {original_shape} -> {filtered_activations.shape}")

                # Save fold-specific extraction file
                fold_extraction_file = output_dir / f"{demographic}_{args.probe_type}_{args.run_id}_fold{fold_idx + 1}_extractions.pkl"

                extraction_data = {
                    'demographic': demographic,
                    'probe_type': args.probe_type,
                    'model': args.model,
                    'run_id': args.run_id,
                    'fold_index': fold_idx,
                    'fold_total': args.n_folds,
                    'train_questions': train_questions,
                    'test_questions': test_questions,
                    'n_samples_per_category': args.n_samples_per_category,
                    'top_k': args.top_k_heads,
                    'top_indices': top_indices,
                    'probing_results': probing_data['probing_results'],
                    'intervention_weights': probing_data['intervention_weights'],
                    'question_extractions': fold_question_extractions,
                    'timestamp': datetime.now().isoformat()
                }

                with open(fold_extraction_file, 'wb') as f:
                    pickle.dump(extraction_data, f)

                print(f"\nSaved fold {fold_idx + 1} extractions: {fold_extraction_file}")

            print(f"\n{'='*80}")
            print(f"COMPLETED K-FOLD EXTRACTION FOR {demographic.upper()}")
            print(f"Saved {args.n_folds} fold-specific extraction files")
            print(f"{'='*80}")

    print("\n" + "="*80)
    print("EXTRACTION PHASE COMPLETE")
    print("="*80)


# ============================================================================
# PHASE 2: K-FOLD INTERVENTION
# ============================================================================

def probe_mlp_layers(activations: torch.Tensor, labels: np.ndarray, num_layers: int, ridge_alpha: float):
    """Probe MLP layers for classification"""
    from sklearn.linear_model import RidgeClassifier
    from sklearn.model_selection import cross_val_score
    from scipy.stats import spearmanr

    layer_results = []

    for layer in range(num_layers):
        layer_act = activations[:, layer, :].numpy()
        clf = RidgeClassifier(alpha=ridge_alpha, random_state=42)

        scores = cross_val_score(clf, layer_act, labels, cv=2, scoring='accuracy')
        clf.fit(layer_act, labels)
        predictions = clf.predict(layer_act)
        spearman_r, p_value = spearmanr(labels, predictions)

        layer_results.append({
            'layer': layer,
            'accuracy': scores.mean(),
            'spearman_r': spearman_r,
            'p_value': p_value,
            'coefficients': clf.coef_ if hasattr(clf, 'coef_') else None
        })

    layer_results = sorted(layer_results, key=lambda x: abs(x['spearman_r']), reverse=True)
    return {'layer_results': layer_results}


def extract_mlp_intervention_weights(results, top_k=10):
    """Extract intervention weights from MLP probing results"""
    layer_results = results['layer_results'][:top_k]
    intervention_weights = {}

    for result in layer_results:
        layer = result['layer']
        coef = result['coefficients']

        # For multiclass: coef is [n_classes, n_features]
        # For binary: coef is [n_features] or [1, n_features]
        # Ensure binary case is 1D
        if coef is not None and coef.ndim > 1 and coef.shape[0] == 1:
            coef = coef[0]

        # Compute std (for multiclass, compute per-class then average)
        if coef is not None:
            if coef.ndim > 1:
                std = np.mean([np.std(coef[i]) for i in range(coef.shape[0])])
            else:
                std = np.std(coef)
        else:
            std = 0.0

        intercept = 0.0
        # Store full coefficient matrix (1D for binary, 2D for multiclass)
        intervention_weights[layer] = (coef, intercept, std)

    return intervention_weights


def train_probes_on_fold(
    activations,
    labels,
    model_config,
    probe_type: str,
    ridge_alpha: float,
    top_k: int
) -> Dict:
    """Train probes on training fold and extract intervention weights"""

    if probe_type == 'attention':
        prober = AttentionHeadProber(
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            head_dim=model_config['head_dim'],
            alpha=ridge_alpha,
            n_folds=2,
            task_type='classification',
            random_state=42
        )

        probing_results = prober.probe_all_heads(activations, labels, aggregation='mean')
        intervention_weights = prober.get_intervention_weights(probing_results, top_k=top_k)

    elif probe_type == 'mlp':
        probing_results = probe_mlp_layers(
            activations, labels, model_config['num_layers'], ridge_alpha
        )
        intervention_weights = extract_mlp_intervention_weights(probing_results, top_k=top_k)

    else:
        raise ValueError(f"Unsupported probe_type for intervention: {probe_type}")

    return intervention_weights


def train_weights_on_prefiltered_activations(
    activations,
    labels,
    top_indices: List,
    probe_type: str,
    ridge_alpha: float,
    top_k: int = None
) -> Dict:
    """
    Train intervention weights on already-filtered activations.

    Activations are assumed to be shape [batch, num_saved_components, dim] where
    components have already been selected in the extraction phase.

    Args:
        top_k: Number of top components to use (default: use all saved components).
               Allows testing with fewer components than were saved during extraction.

    Returns intervention_weights dict mapping component index to (coef, intercept, std).
    """
    from sklearn.linear_model import RidgeClassifier

    intervention_weights = {}

    # Determine how many components to use
    num_saved_components = activations.shape[1]
    if top_k is None:
        top_k = num_saved_components
    else:
        top_k = min(top_k, num_saved_components)

    print(f"  Using top {top_k} of {num_saved_components} saved components")

    # Train a classifier for each of the top_k components
    for component_idx in range(top_k):
        # Get activations for this component: [batch, dim]
        component_activations = activations[:, component_idx, :].numpy()

        # Train ridge classifier
        clf = RidgeClassifier(alpha=ridge_alpha, random_state=42)
        clf.fit(component_activations, labels)

        # Extract weights
        coef = clf.coef_
        # For multiclass: coef is [n_classes, n_features]
        # For binary: coef is [n_features] or [1, n_features]
        # Ensure binary case is 1D
        if coef.ndim > 1 and coef.shape[0] == 1:
            coef = coef[0]

        intercept = clf.intercept_ if hasattr(clf, 'intercept_') else np.array([0.0])
        if not isinstance(intercept, np.ndarray):
            intercept = np.array([intercept])

        # Compute std (for multiclass, compute per-class then average)
        if coef.ndim > 1:
            std = np.mean([np.std(coef[i]) for i in range(coef.shape[0])])
        else:
            std = np.std(coef)

        # Map back to original component index
        original_idx = top_indices[component_idx]
        # Store full coefficient matrix (1D for binary, 2D for multiclass)
        intervention_weights[original_idx] = (coef, intercept, std)

    return intervention_weights


def predict_from_logits(logits: torch.Tensor, answer_options: List, tokenizer) -> str:
    """Get prediction from logits"""
    option_logits = []
    for option in answer_options:
        option_str = str(option)
        tokens = tokenizer.encode(f" {option_str}", add_special_tokens=False)
        if len(tokens) > 0:
            option_logits.append(logits[0, tokens[0]].item())
        else:
            option_logits.append(float('-inf'))

    predicted_idx = np.argmax(option_logits)
    return answer_options[predicted_idx]


def select_class_specific_weights(
    intervention_weights: Dict,
    category_idx: int
) -> Dict:
    """
    Extract class-specific coefficients for multiclass intervention.

    For binary classification, returns weights unchanged.
    For multiclass, selects the coefficient row for the target category.
    """
    class_specific_weights = {}

    for key, (coef, intercept, std) in intervention_weights.items():
        if coef.ndim > 1:
            # Multiclass: select the row for this category
            class_coef = coef[category_idx]
        else:
            # Binary: use as-is
            class_coef = coef

        class_specific_weights[key] = (class_coef, intercept, std)

    return class_specific_weights


def evaluate_intervention_on_fold(
    model,
    tokenizer,
    df: pd.DataFrame,
    test_questions: List[str],
    demographic_attr: str,
    category_names: List[str],
    intervention_weights: Dict,
    device: str,
    probe_type: str,
    intervention_strength: float,
    eval_sample_size: int
) -> Dict:
    """Evaluate intervention on test fold"""

    # Determine config class and parameter name
    if probe_type == 'attention':
        ConfigClass = InterventionConfig
        config_param = 'top_k_heads'
        EngineClass = CircuitInterventionEngine
    else:  # mlp
        ConfigClass = MLPInterventionConfig
        config_param = 'top_k_layers'
        EngineClass = MLPInterventionEngine

    test_results = {}

    for question in test_questions:
        # Get test users
        test_users = df[df[question].notna() & df[demographic_attr].notna()].copy()

        if len(test_users) == 0:
            continue

        if eval_sample_size and len(test_users) > eval_sample_size:
            test_users = test_users.sample(n=eval_sample_size, random_state=42)

        # Get answer options
        answer_options = sorted(test_users[question].dropna().unique().tolist())

        if len(answer_options) == 0:
            continue

        # Print sample size for this question
        q_label = ANES_2024_VARIABLES.get(question, {}).get('label', question)
        print(f"    Testing {question}: {len(test_users)} samples, {len(answer_options)} answer options")
        print(f"      {q_label}")

        baseline_predictions = []
        intervention_predictions = []
        true_labels = []

        # Group users by category for efficient batch processing
        for category_idx, category in enumerate(category_names):
            # Filter users in this category
            category_users = test_users[test_users[demographic_attr] == category]

            if len(category_users) == 0:
                continue

            # Create class-specific intervention weights for this category
            category_weights = select_class_specific_weights(intervention_weights, category_idx)

            # Create intervention engine with class-specific weights
            engine = EngineClass(model, category_weights, device)

            # Determine intervention direction based on binary vs multiclass
            if len(category_names) == 2:
                # Binary classification: use opposite directions for the two classes
                # Category 0: maximize, Category 1: minimize (opposite sides of decision boundary)
                intervention_direction = 'maximize' if category_idx == 0 else 'minimize'
            else:
                # Multiclass: always maximize toward the class-specific direction
                intervention_direction = 'maximize'

            # Config for this category
            config_kwargs = {
                'intervention_strength': intervention_strength,
                config_param: len(category_weights),
                'intervention_direction': intervention_direction
            }
            config = ConfigClass(**config_kwargs)

            # Process all users in this category
            for idx, user_profile in category_users.iterrows():
                prompt = create_prompt(user_profile, question, answer_options=answer_options, answer=None)

                # Baseline prediction
                with torch.no_grad():
                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    baseline_logits = outputs.logits[:, -1, :].cpu()

                baseline_pred = predict_from_logits(baseline_logits, answer_options, tokenizer)
                baseline_predictions.append(baseline_pred)

                # Intervention prediction with class-specific direction
                intervention_logits = engine.intervene_activation_steering_logits(
                    prompt, tokenizer, config
                )

                intervention_pred = predict_from_logits(intervention_logits.cpu(), answer_options, tokenizer)
                intervention_predictions.append(intervention_pred)

                true_labels.append(user_profile[question])

        # Print response distributions
        print(f"\n      Response Distributions:")
        true_dist = Counter(true_labels)
        baseline_dist = Counter(baseline_predictions)
        intervention_dist = Counter(intervention_predictions)

        print(f"        True labels:        {dict(true_dist)}")
        print(f"        Baseline preds:     {dict(baseline_dist)}")
        print(f"        Intervention preds: {dict(intervention_dist)}")

        # Check for prediction collapse
        if len(baseline_dist) == 1:
            print(f"Baseline predictions collapsed to single value!")
        if len(intervention_dist) == 1:
            print(f"Intervention predictions collapsed to single value!")

        # Calculate accuracies
        baseline_acc = accuracy_score(true_labels, baseline_predictions)
        intervention_acc = accuracy_score(true_labels, intervention_predictions)
        improvement = (intervention_acc - baseline_acc) * 100

        # Calculate Kendall's tau for ordinal variables
        # This measures how well predictions preserve the ordinal ranking
        try:
            baseline_kendall, baseline_kendall_p = kendalltau(true_labels, baseline_predictions)
            intervention_kendall, intervention_kendall_p = kendalltau(true_labels, intervention_predictions)

            # Handle NaN values (occurs when all predictions are identical)
            if np.isnan(baseline_kendall):
                baseline_kendall = baseline_kendall_p = None
            if np.isnan(intervention_kendall):
                intervention_kendall = intervention_kendall_p = None

            # Calculate improvement only if both are valid
            if baseline_kendall is not None and intervention_kendall is not None:
                kendall_improvement = intervention_kendall - baseline_kendall
            else:
                kendall_improvement = None

        except (ValueError, TypeError):
            # Handle non-numeric or non-comparable data
            baseline_kendall = baseline_kendall_p = None
            intervention_kendall = intervention_kendall_p = None
            kendall_improvement = None

        test_results[question] = {
            'baseline_accuracy': baseline_acc,
            'intervention_accuracy': intervention_acc,
            'improvement': improvement,
            'baseline_kendall_tau': baseline_kendall,
            'baseline_kendall_p': baseline_kendall_p,
            'intervention_kendall_tau': intervention_kendall,
            'intervention_kendall_p': intervention_kendall_p,
            'kendall_improvement': kendall_improvement,
            'n_samples': len(test_users)
        }

    return test_results


def find_extraction_file(extraction_dir: Path, demographic: str, probe_type: str, run_id: str = None):
    """Find the extraction file for a demographic. If run_id not specified, find most recent."""
    if run_id:
        # Try specific run_id first
        extraction_file = extraction_dir / f"{demographic}_{probe_type}_{run_id}_extractions.pkl"
        if extraction_file.exists():
            return extraction_file

    # Find all matching extraction files
    pattern = f"{demographic}_{probe_type}_*_extractions.pkl"
    matching_files = list(extraction_dir.glob(pattern))

    if not matching_files:
        # Try old format without run_id for backward compatibility
        old_format_file = extraction_dir / f"{demographic}_{probe_type}_extractions.pkl"
        if old_format_file.exists():
            return old_format_file
        return None

    # Return the most recent file based on modification time
    most_recent = max(matching_files, key=lambda p: p.stat().st_mtime)
    return most_recent


def run_intervention_phase(args):
    """
    Phase 2: Load extractions and run k-fold validation on interventions.

    Questions are split into k folds. For each fold:
    - Train probes on k-1 folds
    - Test interventions on remaining fold
    - Aggregate results across folds
    """
    print("\n" + "="*80)
    print("PHASE 2: K-FOLD INTERVENTION")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Probe type: {args.probe_type}")
    print(f"Demographics: {args.demographics}")
    print(f"K-folds: {args.n_folds}")
    print(f"Top-K: {args.top_k_heads}")
    print(f"Intervention strength: {args.intervention_strength}")
    print(f"Run ID: {args.run_id}")
    print("="*80 + "\n")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup directories
    extraction_dir = Path(args.output_dir) / 'extractions'
    results_dir = Path(args.output_dir) / 'intervention_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading ANES data...")
    df = load_anes_data(args.anes_data_path)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on device: {args.device}")

    # Get model config
    model_config = {
        'num_layers': model.config.num_hidden_layers,
        'num_heads': model.config.num_attention_heads,
        'head_dim': model.config.hidden_size // model.config.num_attention_heads,
        'hidden_size': model.config.hidden_size
    }

    # Process each demographic
    all_demographic_results = {}

    for demographic in args.demographics:
        print(f"\n{'='*80}")
        print(f"INTERVENING: {demographic.upper()}")
        print(f"{'='*80}")

        # Check if fold-specific extraction files exist
        fold_files = []
        for fold_idx in range(args.n_folds):
            fold_file = extraction_dir / f"{demographic}_{args.probe_type}_{args.run_id}_fold{fold_idx + 1}_extractions.pkl"
            if fold_file.exists():
                fold_files.append(fold_file)

        # Determine if using fold-specific or global extraction
        using_fold_specific = len(fold_files) == args.n_folds

        if using_fold_specific:
            print(f"Found {len(fold_files)} fold-specific extraction files")
            print(f"Using fold-specific extractions (no data leakage)")
        else:
            # Fall back to global extraction file (old behavior)
            print(f"No fold-specific files found, looking for global extraction...")
            extraction_file = find_extraction_file(extraction_dir, demographic, args.probe_type, args.run_id)
            if extraction_file is None:
                print(f"ERROR: No extraction file found for {demographic} with probe_type={args.probe_type}")
                if args.run_id:
                    print(f"  Looked for run_id: {args.run_id}")
                print("Run extraction phase first!")
                continue

            print(f"Loading global extractions from: {extraction_file}")
            print(f"WARNING: Using global extraction may have data leakage in component selection")
            with open(extraction_file, 'rb') as f:
                global_extraction_data = pickle.load(f)

        fold_results = []

        for fold_idx in range(args.n_folds):
            if using_fold_specific:
                # Load fold-specific extraction
                fold_file = fold_files[fold_idx]
                print(f"\n{'-'*80}")
                print(f"FOLD {fold_idx + 1}/{args.n_folds}")
                print(f"{'-'*80}")
                print(f"Loading fold-specific extraction: {fold_file.name}")

                with open(fold_file, 'rb') as f:
                    extraction_data = pickle.load(f)

                question_extractions = extraction_data['question_extractions']
                train_questions = extraction_data['train_questions']
                test_questions = extraction_data['test_questions']
                top_indices = extraction_data['top_indices']

                print(f"Train questions ({len(train_questions)}): {train_questions}")
                print(f"Test questions ({len(test_questions)}): {test_questions}")
                print(f"Pre-selected top {len(top_indices)} components: {top_indices}")

                # Validate that requested top_k_heads doesn't exceed saved components
                if args.top_k_heads > len(top_indices):
                    print(f"  WARNING: Requested top_k_heads={args.top_k_heads} but only {len(top_indices)} components were saved.")
                    print(f"  Will use all {len(top_indices)} saved components instead.")
                    print(f"  To use more components, re-run extraction phase with larger --top_k_heads")

                has_preselected_components = True
            else:
                # Use global extraction with k-fold splitting (old behavior)
                print(f"\n{'-'*80}")
                print(f"FOLD {fold_idx + 1}/{args.n_folds}")
                print(f"{'-'*80}")

                question_extractions = global_extraction_data['question_extractions']
                questions = list(question_extractions.keys())

                # Setup k-fold cross-validation
                kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
                train_indices, test_indices = list(kfold.split(questions))[fold_idx]

                train_questions = [questions[i] for i in train_indices]
                test_questions = [questions[i] for i in test_indices]

                print(f"Train questions ({len(train_questions)}): {train_questions}")
                print(f"Test questions ({len(test_questions)}): {test_questions}")

                # Check if extraction has pre-selected top components
                has_preselected_components = ('top_indices' in global_extraction_data and
                                             global_extraction_data['top_indices'] is not None)

                if has_preselected_components:
                    top_indices = global_extraction_data['top_indices']
                    print(f"Using global pre-selected top {len(top_indices)} components")

                    # Validate that requested top_k_heads doesn't exceed saved components
                    if args.top_k_heads > len(top_indices):
                        print(f"  WARNING: Requested top_k_heads={args.top_k_heads} but only {len(top_indices)} components were saved.")
                        print(f"  Will use all {len(top_indices)} saved components instead.")
                        print(f"  To use more components, re-run extraction phase with larger --top_k_heads")
                else:
                    print(f"No pre-selected components found, will train probes on this fold")
                    top_indices = None

            # Aggregate training data
            train_activations = []
            train_labels = []
            category_names = None

            for question in train_questions:
                q_data = question_extractions[question]
                activations = q_data['activations']
                labels = q_data['labels']

                if category_names is None:
                    category_names = q_data['category_names']

                if args.probe_type == 'both':
                    # Not supported for intervention
                    print(f"  WARNING: probe_type='both' not supported for intervention")
                    break

                train_activations.append(activations)
                train_labels.append(labels)

            if args.probe_type == 'both':
                continue

            # Concatenate training data
            train_activations = torch.cat(train_activations, dim=0)
            train_labels = np.concatenate(train_labels)

            print(f"\nTraining data shape: {train_activations.shape}")
            print(f"Training labels: {len(train_labels)}")

            # Train intervention weights
            if has_preselected_components:
                # Activations are already filtered, just train weights on pre-selected components
                print(f"\nTraining weights on pre-selected {len(top_indices)} components (will use top {args.top_k_heads})...")
                intervention_weights = train_weights_on_prefiltered_activations(
                    train_activations, train_labels, top_indices,
                    args.probe_type, args.ridge_alpha, top_k=args.top_k_heads
                )
            else:
                # Full probing: select top components and train weights
                print(f"\nTraining probes on {len(train_questions)} questions...")
                intervention_weights = train_probes_on_fold(
                    train_activations, train_labels, model_config,
                    args.probe_type, args.ridge_alpha, args.top_k_heads
                )

            print(f"Extracted {len(intervention_weights)} intervention weights")

            # Evaluate on test fold
            print(f"\nEvaluating on {len(test_questions)} test questions...")
            test_results = evaluate_intervention_on_fold(
                model, tokenizer, df, test_questions, demographic,
                category_names, intervention_weights, args.device,
                args.probe_type, args.intervention_strength, args.eval_sample_size
            )

            # Print fold results
            if test_results:
                fold_baseline = np.mean([r['baseline_accuracy'] for r in test_results.values()])
                fold_intervention = np.mean([r['intervention_accuracy'] for r in test_results.values()])
                fold_improvement = np.mean([r['improvement'] for r in test_results.values()])

                # Calculate Kendall's tau averages (filter out None and NaN values)
                baseline_kendalls = [r['baseline_kendall_tau'] for r in test_results.values()
                                    if r['baseline_kendall_tau'] is not None and not (isinstance(r['baseline_kendall_tau'], float) and np.isnan(r['baseline_kendall_tau']))]
                intervention_kendalls = [r['intervention_kendall_tau'] for r in test_results.values()
                                        if r['intervention_kendall_tau'] is not None and not (isinstance(r['intervention_kendall_tau'], float) and np.isnan(r['intervention_kendall_tau']))]
                kendall_improvements = [r['kendall_improvement'] for r in test_results.values()
                                       if r['kendall_improvement'] is not None and not (isinstance(r['kendall_improvement'], float) and np.isnan(r['kendall_improvement']))]

                fold_baseline_kendall = np.mean(baseline_kendalls) if baseline_kendalls else None
                fold_intervention_kendall = np.mean(intervention_kendalls) if intervention_kendalls else None
                fold_kendall_improvement = np.mean(kendall_improvements) if kendall_improvements else None

                print(f"\nFold {fold_idx + 1} Results:")
                print(f"  Baseline:     {fold_baseline*100:.1f}%")
                print(f"  Intervention: {fold_intervention*100:.1f}%")
                print(f"  Improvement:  {fold_improvement:+.1f} points")
                if fold_baseline_kendall is not None or fold_intervention_kendall is not None:
                    baseline_str = f"{fold_baseline_kendall:.3f}" if fold_baseline_kendall is not None else "N/A"
                    intervention_str = f"{fold_intervention_kendall:.3f}" if fold_intervention_kendall is not None else "N/A"
                    improvement_str = f"{fold_kendall_improvement:+.3f}" if fold_kendall_improvement is not None else "N/A"
                    print(f"  Kendall's tau (Baseline):     {baseline_str}")
                    print(f"  Kendall's tau (Intervention): {intervention_str}")
                    print(f"  Kendall's tau Improvement:    {improvement_str}")

                # Store fold-level aggregates for cross-fold statistics
                fold_aggregate_metrics = {
                    'baseline_accuracy': fold_baseline,
                    'intervention_accuracy': fold_intervention,
                    'improvement': fold_improvement,
                    'baseline_kendall_tau': fold_baseline_kendall,
                    'intervention_kendall_tau': fold_intervention_kendall,
                    'kendall_improvement': fold_kendall_improvement,
                    'n_test_questions': len(test_results)
                }
            else:
                fold_aggregate_metrics = None

            fold_results.append({
                'fold': fold_idx,
                'train_questions': train_questions,
                'test_questions': test_questions,
                'test_results': test_results,
                'intervention_weights': intervention_weights,
                'aggregate_metrics': fold_aggregate_metrics
            })

        # Aggregate results across folds
        print(f"\n{'='*80}")
        print(f"AGGREGATING RESULTS FOR {demographic.upper()}")
        print(f"{'='*80}")

        # Calculate per-question metrics across folds
        all_question_results = {}
        for fold_data in fold_results:
            for question, result in fold_data['test_results'].items():
                if question not in all_question_results:
                    all_question_results[question] = []
                all_question_results[question].append(result)

        # Per-question results (each question tested in exactly one fold)
        question_aggregates = {}
        for question, results_list in all_question_results.items():
            # Each question appears in test set of exactly one fold
            assert len(results_list) == 1, f"Question '{question}' appears in {len(results_list)} folds (expected 1)"
            result = results_list[0]
            question_aggregates[question] = {
                'baseline_accuracy': result['baseline_accuracy'],
                'intervention_accuracy': result['intervention_accuracy'],
                'improvement': result['improvement'],
                'baseline_kendall_tau': result['baseline_kendall_tau'],
                'baseline_kendall_p': result['baseline_kendall_p'],
                'intervention_kendall_tau': result['intervention_kendall_tau'],
                'intervention_kendall_p': result['intervention_kendall_p'],
                'kendall_improvement': result['kendall_improvement'],
                'n_samples': result['n_samples']
            }

        # Aggregate ACROSS FOLDS (correct approach for computing std)
        # Extract per-fold aggregates (mean across test questions in each fold)
        fold_baseline_accs = []
        fold_intervention_accs = []
        fold_improvements = []
        fold_baseline_kendalls = []
        fold_intervention_kendalls = []
        fold_kendall_improvements = []

        for fold_data in fold_results:
            if fold_data['aggregate_metrics'] is not None:
                fold_baseline_accs.append(fold_data['aggregate_metrics']['baseline_accuracy'])
                fold_intervention_accs.append(fold_data['aggregate_metrics']['intervention_accuracy'])
                fold_improvements.append(fold_data['aggregate_metrics']['improvement'])

                # Add Kendall metrics if available
                if fold_data['aggregate_metrics']['baseline_kendall_tau'] is not None:
                    fold_baseline_kendalls.append(fold_data['aggregate_metrics']['baseline_kendall_tau'])
                    fold_intervention_kendalls.append(fold_data['aggregate_metrics']['intervention_kendall_tau'])
                    fold_kendall_improvements.append(fold_data['aggregate_metrics']['kendall_improvement'])

        n_folds = len(fold_baseline_accs)
        overall_baseline_mean = np.mean(fold_baseline_accs)
        overall_baseline_std = np.std(fold_baseline_accs, ddof=1) if n_folds > 1 else None

        overall_intervention_mean = np.mean(fold_intervention_accs)
        overall_intervention_std = np.std(fold_intervention_accs, ddof=1) if n_folds > 1 else None

        overall_improvement_mean = np.mean(fold_improvements)
        overall_improvement_std = np.std(fold_improvements, ddof=1) if n_folds > 1 else None

        # Kendall's tau statistics
        n_kendall_folds = len(fold_baseline_kendalls)
        if n_kendall_folds > 0:
            overall_baseline_kendall_mean = np.mean(fold_baseline_kendalls)
            overall_baseline_kendall_std = np.std(fold_baseline_kendalls, ddof=1) if n_kendall_folds > 1 else None

            overall_intervention_kendall_mean = np.mean(fold_intervention_kendalls)
            overall_intervention_kendall_std = np.std(fold_intervention_kendalls, ddof=1) if n_kendall_folds > 1 else None

            overall_kendall_improvement_mean = np.mean(fold_kendall_improvements)
            overall_kendall_improvement_std = np.std(fold_kendall_improvements, ddof=1) if n_kendall_folds > 1 else None
        else:
            overall_baseline_kendall_mean = overall_baseline_kendall_std = None
            overall_intervention_kendall_mean = overall_intervention_kendall_std = None
            overall_kendall_improvement_mean = overall_kendall_improvement_std = None

        print(f"\nOverall Results (mean ± std across {args.n_folds} folds):")
        if overall_baseline_std is not None:
            print(f"  Baseline:     {overall_baseline_mean*100:.1f}% ± {overall_baseline_std*100:.1f}%")
            print(f"  Intervention: {overall_intervention_mean*100:.1f}% ± {overall_intervention_std*100:.1f}%")
            print(f"  Improvement:  {overall_improvement_mean:+.1f} ± {overall_improvement_std:.1f} points")
            if overall_baseline_kendall_mean is not None:
                kendall_std_str = f" ± {overall_baseline_kendall_std:.3f}" if overall_baseline_kendall_std is not None else ""
                print(f"  Kendall's tau (Baseline):     {overall_baseline_kendall_mean:.3f}{kendall_std_str}")
                kendall_std_str = f" ± {overall_intervention_kendall_std:.3f}" if overall_intervention_kendall_std is not None else ""
                print(f"  Kendall's tau (Intervention): {overall_intervention_kendall_mean:.3f}{kendall_std_str}")
                kendall_std_str = f" ± {overall_kendall_improvement_std:.3f}" if overall_kendall_improvement_std is not None else ""
                print(f"  Kendall's tau Improvement:    {overall_kendall_improvement_mean:+.3f}{kendall_std_str}")
        else:
            print(f"  Baseline:     {overall_baseline_mean*100:.1f}%")
            print(f"  Intervention: {overall_intervention_mean*100:.1f}%")
            print(f"  Improvement:  {overall_improvement_mean:+.1f} points")
            if overall_baseline_kendall_mean is not None:
                print(f"  Kendall's tau (Baseline):     {overall_baseline_kendall_mean:.3f}")
                print(f"  Kendall's tau (Intervention): {overall_intervention_kendall_mean:.3f}")
                print(f"  Kendall's tau Improvement:    {overall_kendall_improvement_mean:+.3f}")
            print(f"  (Single fold: no std available)")

        # Save demographic results
        demographic_results = {
            'demographic': demographic,
            'category_names': category_names,
            'probe_type': args.probe_type,
            'run_id': args.run_id,
            'n_folds': args.n_folds,
            'top_k_heads': args.top_k_heads,
            'intervention_strength': args.intervention_strength,
            'fold_results': fold_results,
            'question_aggregates': question_aggregates,
            'overall_metrics': {
                'baseline_accuracy_mean': overall_baseline_mean,
                'baseline_accuracy_std': overall_baseline_std,
                'intervention_accuracy_mean': overall_intervention_mean,
                'intervention_accuracy_std': overall_intervention_std,
                'improvement_mean': overall_improvement_mean,
                'improvement_std': overall_improvement_std,
                'baseline_kendall_tau_mean': overall_baseline_kendall_mean,
                'baseline_kendall_tau_std': overall_baseline_kendall_std,
                'intervention_kendall_tau_mean': overall_intervention_kendall_mean,
                'intervention_kendall_tau_std': overall_intervention_kendall_std,
                'kendall_improvement_mean': overall_kendall_improvement_mean,
                'kendall_improvement_std': overall_kendall_improvement_std,
                'per_fold_aggregates': {
                    'baseline': fold_baseline_accs,
                    'intervention': fold_intervention_accs,
                    'improvement': fold_improvements,
                    'baseline_kendall': fold_baseline_kendalls,
                    'intervention_kendall': fold_intervention_kendalls,
                    'kendall_improvement': fold_kendall_improvements
                },
                'n_folds': n_folds,
                'n_kendall_folds': n_kendall_folds
            },
            'timestamp': datetime.now().isoformat()
        }

        result_file = results_dir / f"{demographic}_{args.probe_type}_{args.run_id}_intervention_results.pkl"
        with open(result_file, 'wb') as f:
            pickle.dump(demographic_results, f)

        print(f"\nSaved results: {result_file}")

        all_demographic_results[demographic] = demographic_results

    # Save aggregate results
    print(f"\n{'='*80}")
    print("INTERVENTION PHASE COMPLETE")
    print(f"{'='*80}")

    summary_file = results_dir / 'intervention_summary.json'
    summary_data = {
        'config': vars(args),
        'demographics': {
            demo: {
                'baseline_accuracy_mean': results['overall_metrics']['baseline_accuracy_mean'],
                'baseline_accuracy_std': results['overall_metrics']['baseline_accuracy_std'],
                'intervention_accuracy_mean': results['overall_metrics']['intervention_accuracy_mean'],
                'intervention_accuracy_std': results['overall_metrics']['intervention_accuracy_std'],
                'improvement_mean': results['overall_metrics']['improvement_mean'],
                'improvement_std': results['overall_metrics']['improvement_std'],
                'baseline_kendall_tau_mean': results['overall_metrics']['baseline_kendall_tau_mean'],
                'baseline_kendall_tau_std': results['overall_metrics']['baseline_kendall_tau_std'],
                'intervention_kendall_tau_mean': results['overall_metrics']['intervention_kendall_tau_mean'],
                'intervention_kendall_tau_std': results['overall_metrics']['intervention_kendall_tau_std'],
                'kendall_improvement_mean': results['overall_metrics']['kendall_improvement_mean'],
                'kendall_improvement_std': results['overall_metrics']['kendall_improvement_std'],
                'n_folds': results['n_folds'],
                'n_kendall_folds': results['overall_metrics']['n_kendall_folds']
            }
            for demo, results in all_demographic_results.items()
        },
        'timestamp': datetime.now().isoformat()
    }

    # Load existing results if file exists and append new results
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                existing_data = json.load(f)

            # If existing data is a list, append to it
            if isinstance(existing_data, list):
                all_results = existing_data
            # If existing data is a single dict, convert to list
            else:
                all_results = [existing_data]

            all_results.append(summary_data)
            print(f"\nAppending to existing summary (now contains {len(all_results)} runs)")
        except json.JSONDecodeError:
            print(f"\nWarning: Could not parse existing {summary_file}, will overwrite")
            all_results = [summary_data]
    else:
        all_results = [summary_data]

    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSummary saved: {summary_file}")

    # Print final summary
    print(f"\nFinal Summary:")
    for demo, results in all_demographic_results.items():
        metrics = results['overall_metrics']
        print(f"  {demo}: {metrics['baseline_accuracy']*100:.1f}% -> "
              f"{metrics['intervention_accuracy']*100:.1f}% "
              f"({metrics['improvement']:+.1f} points)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    args = parse_arguments()

    print("\n" + "="*80)
    print("ROBUST DEMOGRAPHIC INTERVENTION EXPERIMENT")
    print("="*80)
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Phase: {args.phase}")
    print(f"  Probe type: {args.probe_type}")
    print(f"  Demographics: {args.demographics}")
    print(f"  Run ID: {args.run_id}")
    print(f"  K-folds: {args.n_folds}")
    print(f"  Top-K: {args.top_k_heads}")
    print(f"  Intervention strength: {args.intervention_strength}")
    print(f"  Output directory: {args.output_dir}")
    print("="*80)

    # Run requested phase(s)
    if args.phase in ['extract', 'both']:
        run_extraction_phase(args)

    if args.phase in ['intervene', 'both']:
        run_intervention_phase(args)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
