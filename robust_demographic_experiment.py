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

    # Intersectional intervention (applies multiple demographics simultaneously)
    parser.add_argument('--intersect_demographics', type=str, nargs='+', default=None,
                       help='Demographics to combine for intersectional intervention (e.g., age gender ideology). '
                            'When specified, intervention phase will apply all demographic steering vectors simultaneously '
                            'to test additive effects. Extraction files must exist for all specified demographics.')

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

    # Prompt style
    parser.add_argument('--prompt_style', type=str,
                       choices=['original', 'explicit_instruction', 'first_person', 'chain_of_thought', 'diversity_explicit'],
                       default='original',
                       help='Prompt style to use for baseline predictions (default: original). '
                            'Different styles can reduce model bias and improve prediction diversity.')

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

    # Validate intersectional demographics
    if args.intersect_demographics is not None:
        invalid_intersect = set(args.intersect_demographics) - set(ALL_DEMOGRAPHICS)
        if invalid_intersect:
            parser.error(f"Invalid intersect_demographics: {invalid_intersect}. Valid options: {ALL_DEMOGRAPHICS}")
        if len(args.intersect_demographics) < 2:
            parser.error("intersect_demographics must specify at least 2 demographics to combine")
        if args.phase == 'extract':
            parser.error("intersect_demographics can only be used with --phase intervene or --phase both")

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

def build_demographic_description(user_profile: pd.Series, exclude_attribute: str = None) -> str:
    """
    Build demographic description string, optionally excluding a specific attribute.

    This is used in both extraction (exclude target demographic) and intervention (include all).
    All prompt styles use this function to ensure consistent attribute exclusion.
    """
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

    # Build demographics, excluding the target attribute if specified
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

    return ' '.join(demographic_parts)


def create_prompt(
    user_profile: pd.Series,
    question: str,
    exclude_attribute: str = None,
    answer_options: Optional[List[str]] = None,
    answer: Optional[str] = None,
    prompt_style: str = "original"
) -> str:
    """
    Build a demographic prompt with configurable style.

    Args:
        user_profile: User demographic data
        question: Question ID or text
        exclude_attribute: Attribute to exclude from description (e.g., 'ideology' during extraction).
                          None means include all attributes (default for intervention phase).
        answer_options: List of possible answers
        answer: The actual answer (for training prompts)
        prompt_style: One of ['original', 'explicit_instruction', 'first_person',
                              'chain_of_thought', 'diversity_explicit']

    Returns:
        Formatted prompt string

    Examples:
        # Extraction phase - exclude target demographic
        prompt = create_prompt(profile, "ideology_question", exclude_attribute="ideology")

        # Intervention phase - include all demographics
        prompt = create_prompt(profile, "political_question", prompt_style="explicit_instruction")
    """
    # Build demographic description (respects exclude_attribute)
    demographic = build_demographic_description(user_profile, exclude_attribute)

    # Get question label
    question_label = ANES_2024_VARIABLES.get(question, {}).get('label', question)

    # Build options string
    options_str = ' / '.join(answer_options) if answer_options else ""

    # Build prompt based on style
    if prompt_style == "original":
        # Original simple format
        if answer is not None:
            if answer_options:
                return f"A {demographic} is asked: {question_label} ({options_str}). They answer: {answer}"
            else:
                return f"A {demographic} is asked: {question_label}. They answer: {answer}"
        else:
            if answer_options:
                return f"A {demographic} is asked: {question_label} ({options_str}). They answer:"
            else:
                return f"A {demographic} is asked: {question_label}. They answer:"

    elif prompt_style == "explicit_instruction":
        if answer is not None:
            return f"""Based on the following demographic profile, predict how this person would most likely answer:

Profile: A {demographic}
Question: {question_label}
Options: {options_str}

Consider how their demographic characteristics (age, race, income, ideology, location, etc.) typically influence opinions on this topic.

Answer: {answer}"""
        else:
            return f"""Based on the following demographic profile, predict how this person would most likely answer:

Profile: A {demographic}
Question: {question_label}
Options: {options_str}

Consider how their demographic characteristics (age, race, income, ideology, location, etc.) typically influence opinions on this topic.

Answer:"""

    elif prompt_style == "first_person":
        if answer is not None:
            return f"""I am a {demographic}. I am asked: {question_label}

My answer ({options_str}): {answer}"""
        else:
            return f"""I am a {demographic}. I am asked: {question_label}

My answer ({options_str}):"""

    elif prompt_style == "chain_of_thought":
        # Extract specific attributes for reasoning
        ideology = user_profile.get('ideology', 'unknown')
        income = user_profile.get('income', 'unknown')
        location = user_profile.get('urban_rural', 'unknown')

        if answer is not None:
            return f"""Profile: A {demographic}
Question: {question_label} ({options_str})

Consider:
- How might their {ideology} ideology influence their view?
- How might their {income} income and {location} location affect this?
- What position would be most consistent with their background?

Most likely answer: {answer}"""
        else:
            return f"""Profile: A {demographic}
Question: {question_label} ({options_str})

Consider:
- How might their {ideology} ideology influence their view?
- How might their {income} income and {location} location affect this?
- What position would be most consistent with their background?

Most likely answer:"""

    elif prompt_style == "diversity_explicit":
        if answer is not None:
            return f"""People have diverse opinions based on their backgrounds.
Profile: A {demographic}
Question: {question_label}
Options: {options_str}

What would THIS SPECIFIC person most likely answer, given their unique demographic profile?

Answer: {answer}"""
        else:
            return f"""People have diverse opinions based on their backgrounds.
Profile: A {demographic}
Question: {question_label}
Options: {options_str}

What would THIS SPECIFIC person most likely answer, given their unique demographic profile?

Answer:"""

    else:
        raise ValueError(f"Unknown prompt_style: {prompt_style}. Must be one of: "
                        f"'original', 'explicit_instruction', 'first_person', "
                        f"'chain_of_thought', 'diversity_explicit'")


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
    probe_type: str,
    prompt_style: str = "original"
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
            prompt = create_prompt(
                user_profile, question,
                exclude_attribute=demographic_attr,  # Exclude target demographic
                answer_options=answer_options,
                answer=answer,
                prompt_style=prompt_style
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
    Phase 1: Extract activations for all questions and perform k-fold probing.

    For each fold:
    - Identifies top N components using ONLY training questions (probing)
    - Filters and saves training question activations for those top components
    - Test questions are NOT saved - they will be extracted fresh during intervention

    This approach:
    - Reduces memory/disk usage (only saves training activations)
    - Prevents any data leakage (test questions never influence component selection)
    - Allows interventions to be tested with different configurations without re-extraction
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
                    args.n_samples_per_category, args.device, args.probe_type,
                    args.prompt_style
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

                # Filter ONLY training questions' activations to keep only this fold's top N
                # Note: Test activations are NOT saved here - they will be extracted fresh
                # during the intervention phase when running forward passes with intervention hooks
                print(f"\nFiltering {len(train_questions)} training questions to fold {fold_idx + 1}'s top {args.top_k_heads} components...")
                fold_question_extractions = {}
                for question in train_questions:
                    q_data = question_extractions[question]
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
                        print(f"  {question}: {original_shape} -> {filtered_activations.shape}")

                # Save fold-specific extraction file
                # Note: Only training question activations are saved (filtered to top components)
                # Test questions will be processed fresh during intervention phase
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
                    'question_extractions': fold_question_extractions,  # Contains ONLY train_questions (filtered activations)
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
    """
    Get prediction from logits (simple first-token method).
    DEPRECATED: Use predict_from_logits_multitoken for proper multi-token handling.
    """
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


def predict_from_logits_multitoken(
    model,
    tokenizer,
    prompt: str,
    answer_options: List[str],
    device: str,
    use_intervention: bool = False,
    intervention_engine = None,
    intervention_config = None,
    hooks_already_setup: bool = False
) -> str:
    """
    Get prediction using proper multi-token sequence probability with length normalization.

    Uses batched teacher forcing for efficient parallel computation of all answer options.
    Instead of processing each token autoregressively, we provide the full sequence and
    extract log-probabilities for each token position in a single forward pass.

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt: Full prompt text (context for generation)
        answer_options: List of possible answer strings
        device: Device for computation
        use_intervention: Whether to apply intervention during generation
        intervention_engine: Intervention engine (required if use_intervention=True)
        intervention_config: Intervention config (required if use_intervention=True)
        hooks_already_setup: If True, assumes hooks are already registered (optimization)

    Returns:
        Predicted answer option (string)
    """
    # Tokenize prompt once
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(prompt_tokens)  # Precompute length

    # Tokenize all answer options
    option_strs = [str(option) for option in answer_options]
    option_token_lists = []
    for option_str in option_strs:
        # Add space prefix to match training data format
        tokens = tokenizer.encode(f" {option_str}", add_special_tokens=False)
        if len(tokens) > 0:
            option_token_lists.append((option_str, tokens))

    if len(option_token_lists) == 0:
        return answer_options[0]

    # Prepare batched inputs: prompt + full answer sequence for each option
    batch_input_ids = []
    batch_option_lengths = []

    for option_str, option_tokens in option_token_lists:
        # Concatenate prompt tokens + answer tokens
        full_sequence = prompt_tokens + option_tokens
        batch_input_ids.append(full_sequence)
        batch_option_lengths.append(len(option_tokens))

    # Pad sequences to same length
    max_len = max(len(seq) for seq in batch_input_ids)
    padded_input_ids = []
    attention_masks = []

    for seq in batch_input_ids:
        padding_length = max_len - len(seq)
        # Pad on the left (standard for causal LM)
        padded_seq = [tokenizer.pad_token_id] * padding_length + seq
        mask = [0] * padding_length + [1] * len(seq)

        padded_input_ids.append(padded_seq)
        attention_masks.append(mask)

    # Convert to tensors
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
    attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long).to(device)

    # Get logits with or without intervention
    if use_intervention and intervention_engine is not None:
        # Use batched intervention for maximum performance!
        # Reuse the already-prepared tensors from above (no duplicate work!)

        # Set up intervention hooks (unless already setup for batch processing)
        if not hooks_already_setup:
            intervention_engine._clear_hooks()

            # Determine if we're using attention or MLP engine
            is_attention = isinstance(intervention_engine, CircuitInterventionEngine)

            if is_attention:
                top_k_items = list(intervention_engine.intervention_weights.items())[:intervention_config.top_k_heads]
                for (layer, head), (ridge_coef, intercept, feature_std) in top_k_items:
                    hook = intervention_engine._create_steering_hook(
                        layer, head, ridge_coef, feature_std, intervention_config
                    )
                    intervention_engine.hooks.append(hook)
            else:
                top_k_items = list(intervention_engine.intervention_weights.items())[:intervention_config.top_k_layers]
                for layer_idx, (ridge_coef, intercept, feature_std) in top_k_items:
                    hook = intervention_engine._create_steering_hook(
                        layer_idx, ridge_coef, feature_std, intervention_config
                    )
                    intervention_engine.hooks.append(hook)

        # Single batched forward pass with intervention (FAST!)
        with torch.no_grad():
            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Clear hooks (unless caller wants to reuse them)
        if not hooks_already_setup:
            intervention_engine._clear_hooks()

        # Extract log-probs for each option using teacher forcing
        option_scores = []

        for idx, (option_str, option_tokens) in enumerate(option_token_lists):
            seq_logits = logits[idx]  # (seq_len, vocab_size)

            # Precompute padding offset for this sequence (constant per option)
            padding_length = max_len - len(batch_input_ids[idx])

            # Compute log_softmax ONCE for entire sequence (not per token!)
            seq_log_probs = torch.log_softmax(seq_logits, dim=-1)  # (seq_len, vocab_size)

            # Vectorized extraction: collect all positions and token IDs
            positions = []
            token_ids = []
            for i, token_id in enumerate(option_tokens):
                pos = prompt_len + i - 1
                pos_with_padding = padding_length + pos
                if pos_with_padding >= 0 and pos_with_padding < len(seq_log_probs):
                    positions.append(pos_with_padding)
                    token_ids.append(token_id)

            # Extract all log-probs at once and sum on GPU
            if positions:
                log_prob = seq_log_probs[positions, token_ids].sum().item()  # Single GPU->CPU transfer!
            else:
                log_prob = 0.0

            # Length-normalize
            normalized_score = log_prob / len(option_tokens)
            option_scores.append((normalized_score, option_str))

    else:
        # Baseline: single batched forward pass (MUCH faster!)
        with torch.no_grad():
            outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Compute log-probabilities for each option
        option_scores = []

        for idx, (option_str, option_tokens) in enumerate(option_token_lists):
            # For each token in the answer, get its log-probability
            seq_logits = logits[idx]  # (seq_len, vocab_size)

            # Precompute padding offset for this sequence (constant per option)
            padding_length = max_len - len(batch_input_ids[idx])

            # Compute log_softmax ONCE for entire sequence (not per token!)
            seq_log_probs = torch.log_softmax(seq_logits, dim=-1)  # (seq_len, vocab_size)

            # Vectorized extraction: collect all positions and token IDs
            positions = []
            token_ids = []
            for i, token_id in enumerate(option_tokens):
                pos = prompt_len + i - 1
                pos_with_padding = padding_length + pos
                if pos_with_padding >= 0 and pos_with_padding < len(seq_log_probs):
                    positions.append(pos_with_padding)
                    token_ids.append(token_id)

            # Extract all log-probs at once and sum on GPU
            if positions:
                log_prob = seq_log_probs[positions, token_ids].sum().item()  # Single GPU->CPU transfer!
            else:
                log_prob = 0.0

            # Length-normalize
            normalized_score = log_prob / len(option_tokens)
            option_scores.append((normalized_score, option_str))

    # Select best option
    best_score, best_option = max(option_scores, key=lambda x: x[0])
    return best_option


def select_class_specific_weights(
    intervention_weights: Dict,
    category_idx: int
) -> Dict:
    """
    Extract class-specific coefficients for multiclass intervention.

    For binary classification, negates coefficient for category 0 so both categories
    point toward themselves when using 'maximize' intervention direction.
    For multiclass, selects the coefficient row for the target category.
    """
    class_specific_weights = {}

    for key, (coef, intercept, std) in intervention_weights.items():
        if coef.ndim > 1:
            # Multiclass: select the row for this category
            class_coef = coef[category_idx]
        else:
            # Binary: negate for category 0 so both categories point toward themselves
            # Ridge coefficient points class0→class1, so:
            # - Category 0 needs class1→class0 (negate)
            # - Category 1 needs class0→class1 (keep as-is)
            class_coef = -coef if category_idx == 0 else coef

        class_specific_weights[key] = (class_coef, intercept, std)

    return class_specific_weights


# ============================================================================
# ADVANCED METRICS FOR DISTRIBUTION QUALITY
# ============================================================================

def get_distribution(labels: List) -> Dict:
    """Get probability distribution from list of labels"""
    counts = Counter(labels)
    total = len(labels)
    return {k: v/total for k, v in counts.items()}


def prediction_entropy(predictions: List) -> float:
    """
    Shannon entropy of predictions. Measures diversity.

    Returns:
        0 = all predictions identical (complete collapse)
        log2(K) = uniform distribution (maximum diversity)
    """
    counts = Counter(predictions)
    total = len(predictions)
    probs = [c/total for c in counts.values()]
    from scipy.stats import entropy
    return entropy(probs, base=2)


def gini_diversity(predictions: List) -> float:
    """
    Gini diversity index. Measures prediction diversity.

    Returns:
        0 = all predictions identical
        ~1 = highly diverse predictions
    """
    counts = Counter(predictions)
    total = len(predictions)
    probs = [c/total for c in counts.values()]
    return 1 - sum(p**2 for p in probs)


def js_divergence(true_labels: List, predictions: List) -> float:
    """
    Jensen-Shannon divergence between true and predicted distributions.

    Returns:
        0 = distributions identical (perfect match)
        1 = completely different distributions
    """
    from scipy.spatial.distance import jensenshannon

    # Get all possible categories
    all_categories = sorted(set(true_labels) | set(predictions))

    # Build probability vectors
    true_dist = get_distribution(true_labels)
    pred_dist = get_distribution(predictions)

    true_vec = [true_dist.get(cat, 0) for cat in all_categories]
    pred_vec = [pred_dist.get(cat, 0) for cat in all_categories]

    return jensenshannon(true_vec, pred_vec, base=2)


def total_variation_distance(true_labels: List, predictions: List) -> float:
    """
    Total variation distance between distributions.

    Returns:
        0 = distributions identical
        1 = completely disjoint distributions
    """
    true_dist = get_distribution(true_labels)
    pred_dist = get_distribution(predictions)

    all_categories = set(true_dist.keys()) | set(pred_dist.keys())

    return 0.5 * sum(abs(true_dist.get(cat, 0) - pred_dist.get(cat, 0))
                     for cat in all_categories)


def distribution_quality_score(true_labels: List, predictions: List) -> float:
    """
    Combined metric measuring both diversity and distributional similarity.

    Returns:
        Score between 0-1 where higher is better
        - Rewards prediction diversity
        - Rewards similarity to true distribution
    """
    # Component 1: Normalized diversity (0-1)
    ent = prediction_entropy(predictions)
    n_categories = len(set(true_labels))
    max_entropy = np.log2(n_categories) if n_categories > 1 else 1.0
    normalized_diversity = ent / max_entropy if max_entropy > 0 else 0

    # Component 2: Distributional similarity (0-1)
    jsd = js_divergence(true_labels, predictions)
    similarity = 1 - jsd  # Invert so higher is better

    # Combine with equal weights
    return 0.5 * normalized_diversity + 0.5 * similarity


def compute_advanced_metrics(true_labels: List, predictions: List) -> Dict:
    """
    Compute all advanced metrics for evaluating prediction quality.

    Returns dict with:
        - Diversity metrics (entropy, gini)
        - Distributional similarity (js_divergence, total_variation)
        - Multiclass performance (macro_f1, balanced_accuracy, cohen_kappa)
        - Combined quality score
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score

    metrics = {}

    try:
        # Diversity metrics
        metrics['entropy'] = prediction_entropy(predictions)
        metrics['gini_diversity'] = gini_diversity(predictions)

        # Distributional similarity
        metrics['js_divergence'] = js_divergence(true_labels, predictions)
        metrics['total_variation'] = total_variation_distance(true_labels, predictions)

        # Multiclass performance
        metrics['macro_f1'] = f1_score(true_labels, predictions, average='macro', zero_division=0)
        metrics['balanced_accuracy'] = balanced_accuracy_score(true_labels, predictions)
        metrics['cohen_kappa'] = cohen_kappa_score(true_labels, predictions)

        # Combined quality score
        metrics['dist_quality_score'] = distribution_quality_score(true_labels, predictions)

    except Exception as e:
        # If any metric fails, set all to None
        print(f"      Warning: Could not compute advanced metrics: {e}")
        metrics = {k: None for k in ['entropy', 'gini_diversity', 'js_divergence',
                                      'total_variation', 'macro_f1', 'balanced_accuracy',
                                      'cohen_kappa', 'dist_quality_score']}

    return metrics


def combine_demographic_weights(
    demographic_weights_dict: Dict[str, Dict],
    user_profile: pd.Series,
    demographic_attrs: List[str],
    demographic_categories: Dict[str, List[str]],
    verbose: bool = False
) -> Dict:
    """
    Combine intervention weights from multiple demographics for intersectional intervention.

    This function implements the additive hypothesis: demographic steering vectors are summed
    when they target the same attention head/layer. This tests if demographic circuits are
    independent and additive.

    Args:
        demographic_weights_dict: {demographic_name: intervention_weights}
            Each intervention_weights is Dict[(layer, head)] -> (coef, intercept, std)
        user_profile: User's demographic profile (pd.Series with all attributes)
        demographic_attrs: List of demographics to combine (e.g., ['age', 'gender', 'ideology'])
        demographic_categories: {demographic_name: [category_names]}
            e.g., {'age': ['Young', 'Old'], 'gender': ['Male', 'Female']}

    Returns:
        Combined intervention weights: Dict[(layer, head)] -> (combined_coef, intercept, combined_std)
        - combined_coef: Sum of normalized coefficients from all demographics
        - intercept: Average of intercepts (or could use max)
        - combined_std: Maximum of stds (conservative choice for scaling)
    """
    combined_weights = {}

    # Track components from each demographic for analysis
    component_sources = {}  # (layer, head) -> list of demographics

    for demographic in demographic_attrs:
        # Get user's category for this demographic
        user_category = user_profile[demographic]
        category_names = demographic_categories[demographic]

        if user_category not in category_names:
            print(f"Warning: User's {demographic} value '{user_category}' not in trained categories {category_names}")
            continue

        category_idx = category_names.index(user_category)

        # Get intervention weights for this demographic
        demo_weights = demographic_weights_dict[demographic]

        # Get class-specific weights for user's category
        class_weights = select_class_specific_weights(demo_weights, category_idx)

        # Add/combine weights for each component
        for component_key, (coef, intercept, std) in class_weights.items():
            if component_key in combined_weights:
                # Component already exists from another demographic - sum coefficients
                existing_coef, existing_intercept, existing_std = combined_weights[component_key]

                # Sum the coefficients (additive hypothesis)
                combined_coef = existing_coef + coef

                # Average the intercepts
                combined_intercept = (existing_intercept + intercept) / 2

                # Use maximum std (conservative)
                combined_std = max(existing_std, std)

                combined_weights[component_key] = (combined_coef, combined_intercept, combined_std)
                component_sources[component_key].append(demographic)
            else:
                # First demographic to target this component
                combined_weights[component_key] = (coef, intercept, std)
                component_sources[component_key] = [demographic]

    # Print statistics about component overlap (only if verbose)
    if verbose:
        n_shared = sum(1 for sources in component_sources.values() if len(sources) > 1)
        n_unique = sum(1 for sources in component_sources.values() if len(sources) == 1)
        print(f"      Combined {len(demographic_attrs)} demographics:")
        print(f"        Total components: {len(combined_weights)}")
        print(f"        Shared components: {n_shared}")
        print(f"        Unique components: {n_unique}")

    return combined_weights


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
    eval_sample_size: int,
    prompt_style: str = "original",
    output_dir: Path = None,
    run_id: str = None,
    top_k_heads: int = None
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

    for question in tqdm(test_questions, desc="    Evaluating questions", leave=False):
        # Get test users
        test_users = df[df[question].notna() & df[demographic_attr].notna()].copy()

        if len(test_users) == 0:
            continue

        if eval_sample_size and len(test_users) > eval_sample_size:
            test_users = test_users.sample(n=eval_sample_size, random_state=42)

        # Get answer options in proper ordinal order (not alphabetical!)
        if question in ANES_2024_VARIABLES and 'values' in ANES_2024_VARIABLES[question]:
            # Use ANES ordinal order (preserves 1, 2, 3, 4... ranking)
            anes_values = ANES_2024_VARIABLES[question]['values']
            answer_options = [val for val in anes_values.values() if val in test_users[question].values]

            # Special handling: For Favor/Oppose/Neither questions, put Neither in middle
            # Neither represents the middle ground between Favor and Oppose
            if set(answer_options) == {'Favor', 'Oppose', 'Neither'}:
                answer_options = ['Favor', 'Neither', 'Oppose']  # Correct ordinal order
        else:
            # Fall back to alphabetical for non-ordinal questions
            answer_options = sorted(test_users[question].dropna().unique().tolist())

        if len(answer_options) == 0:
            continue

        # Create label-to-rank mapping for Kendall's tau (ordinal correlation)
        # Maps string labels to numeric ranks: 0, 1, 2, 3...
        label_to_rank = {label: rank for rank, label in enumerate(answer_options)}

        # Print sample size for this question
        q_label = ANES_2024_VARIABLES.get(question, {}).get('label', question)
        print(f"    Testing {question}: {len(test_users)} samples, {len(answer_options)} answer options")
        print(f"      {q_label}")

        baseline_predictions = []
        intervention_predictions = []
        true_labels = []

        # Track detailed per-user information for CSV export
        user_details = []  # List of dicts with user info

        # Create progress bar for samples
        total_samples = len(test_users)
        sample_pbar = tqdm(total=total_samples, desc=f"      Processing samples", leave=False)

        # PHASE 1: Collect all baseline predictions (NO hooks active)
        # This ensures baseline predictions are unaffected by intervention
        user_data_by_category = {}  # Store user data for phase 2

        for category_idx, category in enumerate(category_names):
            # Filter users in this category
            category_users = test_users[test_users[demographic_attr] == category]

            if len(category_users) == 0:
                continue

            # Store for phase 2
            user_data_by_category[category_idx] = {
                'category': category,
                'users': category_users,
                'prompts': [],
                'true_labels_local': []
            }

            # Process all users: baseline only (no hooks registered)
            for idx, user_profile in category_users.iterrows():
                prompt = create_prompt(
                    user_profile, question,
                    exclude_attribute=None,  # Include all demographics for prediction
                    answer_options=answer_options,
                    answer=None,
                    prompt_style=prompt_style
                )

                # Store prompt and label for phase 2
                user_data_by_category[category_idx]['prompts'].append(prompt)
                user_data_by_category[category_idx]['true_labels_local'].append(user_profile[question])

                # Baseline prediction (NO intervention hooks active)
                baseline_pred = predict_from_logits_multitoken(
                    model, tokenizer, prompt, answer_options, device,
                    use_intervention=False
                )
                baseline_predictions.append(baseline_pred)
                true_labels.append(user_profile[question])

                # Track user details for CSV export
                user_details.append({
                    'user_id': idx,
                    'demographic_value': category,
                    'true_label': user_profile[question],
                    'baseline_prediction': baseline_pred
                })

                # Update progress bar
                sample_pbar.update(1)

        # Reset progress bar for phase 2
        sample_pbar.close()
        sample_pbar = tqdm(total=total_samples, desc=f"      Intervention phase", leave=False)

        # PHASE 2: Collect intervention predictions (hooks once per category)
        user_idx = 0  # Track position in user_details list
        for category_idx in sorted(user_data_by_category.keys()):
            category_data = user_data_by_category[category_idx]

            # Create class-specific intervention weights for this category
            category_weights = select_class_specific_weights(intervention_weights, category_idx)

            # Create intervention engine with class-specific weights
            engine = EngineClass(model, category_weights, device)

            # Always maximize - select_class_specific_weights already orients coefficients correctly
            intervention_direction = 'maximize'

            # Config for this category
            config_kwargs = {
                'intervention_strength': intervention_strength,
                config_param: len(category_weights),
                'intervention_direction': intervention_direction
            }
            config = ConfigClass(**config_kwargs)

            # Set up intervention hooks ONCE for this entire category (major optimization!)
            engine._clear_hooks()
            is_attention = isinstance(engine, CircuitInterventionEngine)

            if is_attention:
                top_k_items = list(category_weights.items())[:config.top_k_heads]
                for (layer, head), (ridge_coef, intercept, feature_std) in top_k_items:
                    hook = engine._create_steering_hook(
                        layer, head, ridge_coef, feature_std, config
                    )
                    engine.hooks.append(hook)
            else:
                top_k_items = list(category_weights.items())[:config.top_k_layers]
                for layer_idx, (ridge_coef, intercept, feature_std) in top_k_items:
                    hook = engine._create_steering_hook(
                        layer_idx, ridge_coef, feature_std, config
                    )
                    engine.hooks.append(hook)

            # Process all users in this category with same hooks
            for prompt in category_data['prompts']:
                # Intervention prediction (reuse hooks for all samples in this category!)
                intervention_pred = predict_from_logits_multitoken(
                    model, tokenizer, prompt, answer_options, device,
                    use_intervention=True,
                    intervention_engine=engine,
                    intervention_config=config,
                    hooks_already_setup=True  # Hooks are already set up!
                )
                intervention_predictions.append(intervention_pred)

                # Add intervention prediction to user_details
                user_details[user_idx]['intervention_prediction'] = intervention_pred
                user_idx += 1

                # Update progress bar
                sample_pbar.update(1)

            # Clear hooks after processing all users in this category
            engine._clear_hooks()

        # Close the progress bar
        sample_pbar.close()

        # Print response distributions
        print(f"\n      Response Distributions:")
        true_dist = Counter(true_labels)
        baseline_dist = Counter(baseline_predictions)
        intervention_dist = Counter(intervention_predictions)

        # Sort by answer_options order for consistent display
        sorted_true = {k: true_dist[k] for k in answer_options if k in true_dist}
        sorted_baseline = {k: baseline_dist[k] for k in answer_options if k in baseline_dist}
        sorted_intervention = {k: intervention_dist[k] for k in answer_options if k in intervention_dist}

        print(f"        True labels:        {sorted_true}")
        print(f"        Baseline preds:     {sorted_baseline}")
        print(f"        Intervention preds: {sorted_intervention}")

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
            # Convert string labels to numeric ranks for proper ordinal correlation
            true_ranks = [label_to_rank.get(label, -1) for label in true_labels]
            baseline_ranks = [label_to_rank.get(label, -1) for label in baseline_predictions]
            intervention_ranks = [label_to_rank.get(label, -1) for label in intervention_predictions]

            # Filter valid pairs (exclude any -1 ranks from missing labels)
            baseline_valid_pairs = [(t, b) for t, b in zip(true_ranks, baseline_ranks) if t != -1 and b != -1]
            intervention_valid_pairs = [(t, i) for t, i in zip(true_ranks, intervention_ranks) if t != -1 and i != -1]

            # Compute Kendall's tau on numeric ranks
            if len(baseline_valid_pairs) > 1:
                true_b, pred_b = zip(*baseline_valid_pairs)
                baseline_kendall, baseline_kendall_p = kendalltau(true_b, pred_b)
            else:
                baseline_kendall = baseline_kendall_p = None

            if len(intervention_valid_pairs) > 1:
                true_i, pred_i = zip(*intervention_valid_pairs)
                intervention_kendall, intervention_kendall_p = kendalltau(true_i, pred_i)
            else:
                intervention_kendall = intervention_kendall_p = None

            # Handle NaN values (occurs when all predictions are identical)
            if baseline_kendall is not None and np.isnan(baseline_kendall):
                baseline_kendall = baseline_kendall_p = None
            if intervention_kendall is not None and np.isnan(intervention_kendall):
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

        # Calculate advanced metrics (diversity and distributional quality)
        baseline_advanced = compute_advanced_metrics(true_labels, baseline_predictions)
        intervention_advanced = compute_advanced_metrics(true_labels, intervention_predictions)

        # Calculate improvements for key metrics
        improvements = {}
        for key in baseline_advanced.keys():
            if baseline_advanced[key] is not None and intervention_advanced[key] is not None:
                # For divergence metrics (lower is better), improvement is negative delta
                if key in ['js_divergence', 'total_variation']:
                    improvements[f'{key}_improvement'] = baseline_advanced[key] - intervention_advanced[key]
                else:
                    # For other metrics (higher is better), improvement is positive delta
                    improvements[f'{key}_improvement'] = intervention_advanced[key] - baseline_advanced[key]
            else:
                improvements[f'{key}_improvement'] = None

        # Create predictions DataFrame for CSV export
        predictions_df = pd.DataFrame(user_details)
        predictions_df.insert(0, 'question', question)
        predictions_df.insert(1, 'question_label', q_label)
        predictions_df.insert(3, 'demographic', demographic_attr)
        predictions_df['baseline_correct'] = predictions_df['true_label'] == predictions_df['baseline_prediction']
        predictions_df['intervention_correct'] = predictions_df['true_label'] == predictions_df['intervention_prediction']
        predictions_df['changed'] = predictions_df['baseline_prediction'] != predictions_df['intervention_prediction']

        # Save to CSV (one per question)
        if output_dir is not None and run_id is not None and top_k_heads is not None:
            # Construct filename matching intervention results pattern
            csv_filename = f"{demographic_attr}_{probe_type}_k{top_k_heads}_s{intervention_strength}_{run_id}_{question}_predictions.csv"
            csv_path = output_dir / csv_filename
            predictions_df.to_csv(csv_path, index=False)
            print(f"      Saved predictions to {csv_path}")

        test_results[question] = {
            'baseline_accuracy': baseline_acc,
            'intervention_accuracy': intervention_acc,
            'improvement': improvement,
            'baseline_kendall_tau': baseline_kendall,
            'baseline_kendall_p': baseline_kendall_p,
            'intervention_kendall_tau': intervention_kendall,
            'intervention_kendall_p': intervention_kendall_p,
            'kendall_improvement': kendall_improvement,
            # Advanced baseline metrics
            'baseline_entropy': baseline_advanced['entropy'],
            'baseline_gini_diversity': baseline_advanced['gini_diversity'],
            'baseline_js_divergence': baseline_advanced['js_divergence'],
            'baseline_total_variation': baseline_advanced['total_variation'],
            'baseline_macro_f1': baseline_advanced['macro_f1'],
            'baseline_balanced_accuracy': baseline_advanced['balanced_accuracy'],
            'baseline_cohen_kappa': baseline_advanced['cohen_kappa'],
            'baseline_dist_quality_score': baseline_advanced['dist_quality_score'],
            # Advanced intervention metrics
            'intervention_entropy': intervention_advanced['entropy'],
            'intervention_gini_diversity': intervention_advanced['gini_diversity'],
            'intervention_js_divergence': intervention_advanced['js_divergence'],
            'intervention_total_variation': intervention_advanced['total_variation'],
            'intervention_macro_f1': intervention_advanced['macro_f1'],
            'intervention_balanced_accuracy': intervention_advanced['balanced_accuracy'],
            'intervention_cohen_kappa': intervention_advanced['cohen_kappa'],
            'intervention_dist_quality_score': intervention_advanced['dist_quality_score'],
            # Improvements
            **improvements,
            'n_samples': len(test_users)
        }

    return test_results


def evaluate_intersectional_intervention_on_fold(
    model,
    tokenizer,
    df: pd.DataFrame,
    test_questions: List[str],
    demographic_attrs: List[str],
    demographic_weights_dict: Dict[str, Dict],
    demographic_categories: Dict[str, List[str]],
    device: str,
    probe_type: str,
    intervention_strength: float,
    eval_sample_size: int,
    top_k_heads: int,
    prompt_style: str = "original",
    output_dir: Path = None,
    run_id: str = None
) -> Dict:
    """
    Evaluate intersectional intervention combining multiple demographics simultaneously.

    Args:
        model: Language model
        tokenizer: Tokenizer
        df: ANES dataframe
        test_questions: Questions to test on
        demographic_attrs: List of demographics to combine (e.g., ['age', 'gender', 'ideology'])
        demographic_weights_dict: {demographic_name: intervention_weights}
        demographic_categories: {demographic_name: [category_names]}
        device: Device for computation
        probe_type: 'attention' or 'mlp'
        intervention_strength: Strength multiplier
        eval_sample_size: Max samples per question
        prompt_style: Prompt template style

    Returns:
        Dict of test results per question, including intersectional metadata
    """
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
        # Get test users who have valid values for ALL specified demographics
        valid_mask = df[question].notna()
        for demo in demographic_attrs:
            valid_mask &= df[demo].notna()

        test_users = df[valid_mask].copy()

        if len(test_users) == 0:
            print(f"    WARNING: No users with valid values for all demographics {demographic_attrs} on question {question}")
            continue

        if eval_sample_size and len(test_users) > eval_sample_size:
            test_users = test_users.sample(n=eval_sample_size, random_state=42)

        # Get answer options in proper ordinal order (not alphabetical!)
        if question in ANES_2024_VARIABLES and 'values' in ANES_2024_VARIABLES[question]:
            # Use ANES ordinal order (preserves 1, 2, 3, 4... ranking)
            anes_values = ANES_2024_VARIABLES[question]['values']
            answer_options = [val for val in anes_values.values() if val in test_users[question].values]

            # Special handling: For Favor/Oppose/Neither questions, put Neither in middle
            # Neither represents the middle ground between Favor and Oppose
            if set(answer_options) == {'Favor', 'Oppose', 'Neither'}:
                answer_options = ['Favor', 'Neither', 'Oppose']  # Correct ordinal order
        else:
            # Fall back to alphabetical for non-ordinal questions
            answer_options = sorted(test_users[question].dropna().unique().tolist())

        if len(answer_options) == 0:
            continue

        # Create label-to-rank mapping for Kendall's tau (ordinal correlation)
        # Maps string labels to numeric ranks: 0, 1, 2, 3...
        label_to_rank = {label: rank for rank, label in enumerate(answer_options)}

        # Print sample size for this question
        q_label = ANES_2024_VARIABLES.get(question, {}).get('label', question)
        print(f"    Testing {question}: {len(test_users)} samples, {len(answer_options)} answer options")
        print(f"      {q_label}")

        baseline_predictions = []
        intervention_predictions = []
        true_labels = []
        demographic_combinations = []  # Track demographic profile of each user
        user_details = []  # List of dicts with user info for CSV export

        # Process each user individually (since each has unique demographic combination)
        first_user = True
        for idx, user_profile in test_users.iterrows():
            # Combine weights for this user's specific demographic profile
            # Print stats only for first user of each question
            combined_weights = combine_demographic_weights(
                demographic_weights_dict,
                user_profile,
                demographic_attrs,
                demographic_categories,
                verbose=first_user
            )

            if len(combined_weights) == 0:
                print(f"      Warning: No valid weights for user {idx}")
                first_user = False
                continue

            # Show component info for first user
            if first_user:
                print(f"      Using {len(combined_weights)} combined components from top-{top_k_heads} of each demographic")

                # Show magnitude of combined components to diagnose strength
                sorted_for_display = sorted(
                    combined_weights.items(),
                    key=lambda item: np.linalg.norm(item[1][0]),
                    reverse=True
                )
                print(f"      Top combined component magnitudes:", end="")
                for comp_key, (coef, _, _) in sorted_for_display[:3]:
                    mag = np.linalg.norm(coef)
                    print(f" {mag:.2f}", end="")
                print()

            first_user = False

            # Create intervention engine with combined weights
            # Note: weights are already top-k from each demographic, so we use ALL of them
            engine = EngineClass(model, combined_weights, device)

            # For intersectional, we always use 'maximize' direction
            # (each demographic's direction is already baked into the coefficients)
            intervention_direction = 'maximize'

            # Config: use ALL combined components (they're already filtered to top-k per demographic)
            config_kwargs = {
                'intervention_strength': intervention_strength,
                config_param: len(combined_weights),
                'intervention_direction': intervention_direction
            }
            config = ConfigClass(**config_kwargs)

            # Create prompt (include all demographics)
            prompt = create_prompt(
                user_profile, question,
                exclude_attribute=None,
                answer_options=answer_options,
                answer=None,
                prompt_style=prompt_style
            )

            # Baseline prediction (multi-token with length normalization)
            baseline_pred = predict_from_logits_multitoken(
                model, tokenizer, prompt, answer_options, device,
                use_intervention=False
            )
            baseline_predictions.append(baseline_pred)

            # Intervention prediction (multi-token with intervention applied)
            intervention_pred = predict_from_logits_multitoken(
                model, tokenizer, prompt, answer_options, device,
                use_intervention=True,
                intervention_engine=engine,
                intervention_config=config
            )
            intervention_predictions.append(intervention_pred)

            true_labels.append(user_profile[question])

            # Track demographic combination
            demo_combo = tuple(user_profile[demo] for demo in demographic_attrs)
            demographic_combinations.append(demo_combo)

            # Track user details for CSV export
            user_detail = {
                'user_id': idx,
                'true_label': user_profile[question],
                'baseline_prediction': baseline_pred,
                'intervention_prediction': intervention_pred
            }
            # Add each demographic attribute as a column
            for demo in demographic_attrs:
                user_detail[f'demographic_{demo}'] = user_profile[demo]
            user_details.append(user_detail)

        # Print response distributions
        print(f"\n      Response Distributions:")
        true_dist = Counter(true_labels)
        baseline_dist = Counter(baseline_predictions)
        intervention_dist = Counter(intervention_predictions)

        # Sort by answer_options order for consistent display
        sorted_true = {k: true_dist[k] for k in answer_options if k in true_dist}
        sorted_baseline = {k: baseline_dist[k] for k in answer_options if k in baseline_dist}
        sorted_intervention = {k: intervention_dist[k] for k in answer_options if k in intervention_dist}

        print(f"        True labels:        {sorted_true}")
        print(f"        Baseline preds:     {sorted_baseline}")
        print(f"        Intervention preds: {sorted_intervention}")

        # Calculate accuracies
        baseline_acc = accuracy_score(true_labels, baseline_predictions)
        intervention_acc = accuracy_score(true_labels, intervention_predictions)
        improvement = (intervention_acc - baseline_acc) * 100

        # Calculate Kendall's tau for ordinal variables
        # This measures how well predictions preserve the ordinal ranking
        try:
            # Convert string labels to numeric ranks for proper ordinal correlation
            true_ranks = [label_to_rank.get(label, -1) for label in true_labels]
            baseline_ranks = [label_to_rank.get(label, -1) for label in baseline_predictions]
            intervention_ranks = [label_to_rank.get(label, -1) for label in intervention_predictions]

            # Filter valid pairs (exclude any -1 ranks from missing labels)
            baseline_valid_pairs = [(t, b) for t, b in zip(true_ranks, baseline_ranks) if t != -1 and b != -1]
            intervention_valid_pairs = [(t, i) for t, i in zip(true_ranks, intervention_ranks) if t != -1 and i != -1]

            # Compute Kendall's tau on numeric ranks
            if len(baseline_valid_pairs) > 1:
                true_b, pred_b = zip(*baseline_valid_pairs)
                baseline_kendall, baseline_kendall_p = kendalltau(true_b, pred_b)
            else:
                baseline_kendall = baseline_kendall_p = None

            if len(intervention_valid_pairs) > 1:
                true_i, pred_i = zip(*intervention_valid_pairs)
                intervention_kendall, intervention_kendall_p = kendalltau(true_i, pred_i)
            else:
                intervention_kendall = intervention_kendall_p = None

            # Handle NaN values (occurs when all predictions are identical)
            if baseline_kendall is not None and np.isnan(baseline_kendall):
                baseline_kendall = baseline_kendall_p = None
            if intervention_kendall is not None and np.isnan(intervention_kendall):
                intervention_kendall = intervention_kendall_p = None

            # Calculate improvement only if both are valid
            if baseline_kendall is not None and intervention_kendall is not None:
                kendall_improvement = intervention_kendall - baseline_kendall
            else:
                kendall_improvement = None
        except Exception as e:
            print(f"      Warning: Could not compute Kendall's tau: {e}")
            baseline_kendall = baseline_kendall_p = None
            intervention_kendall = intervention_kendall_p = None
            kendall_improvement = None

        # Calculate advanced metrics
        baseline_advanced = compute_advanced_metrics(true_labels, baseline_predictions)
        intervention_advanced = compute_advanced_metrics(true_labels, intervention_predictions)

        # Calculate improvements for key metrics
        improvements = {}
        for key in baseline_advanced.keys():
            if baseline_advanced[key] is not None and intervention_advanced[key] is not None:
                if key in ['js_divergence', 'total_variation']:
                    improvements[f'{key}_improvement'] = baseline_advanced[key] - intervention_advanced[key]
                else:
                    improvements[f'{key}_improvement'] = intervention_advanced[key] - baseline_advanced[key]
            else:
                improvements[f'{key}_improvement'] = None

        # Create predictions DataFrame for CSV export
        predictions_df = pd.DataFrame(user_details)
        predictions_df.insert(0, 'question', question)
        predictions_df.insert(1, 'question_label', q_label)
        predictions_df['baseline_correct'] = predictions_df['true_label'] == predictions_df['baseline_prediction']
        predictions_df['intervention_correct'] = predictions_df['true_label'] == predictions_df['intervention_prediction']
        predictions_df['changed'] = predictions_df['baseline_prediction'] != predictions_df['intervention_prediction']

        # Save to CSV (one per question)
        if output_dir is not None and run_id is not None:
            # Create filename matching intersectional results pattern
            intersect_name = '_'.join(demographic_attrs)
            csv_filename = f"intersectional_{intersect_name}_{probe_type}_k{top_k_heads}_s{intervention_strength}_{run_id}_{question}_predictions.csv"
            csv_path = output_dir / csv_filename
            predictions_df.to_csv(csv_path, index=False)
            print(f"      Saved predictions to {csv_path}")

        # Count unique demographic combinations tested
        unique_combos = len(set(demographic_combinations))
        combo_dist = Counter(demographic_combinations)

        test_results[question] = {
            'baseline_accuracy': baseline_acc,
            'intervention_accuracy': intervention_acc,
            'improvement': improvement,
            'baseline_kendall_tau': baseline_kendall,
            'baseline_kendall_p': baseline_kendall_p,
            'intervention_kendall_tau': intervention_kendall,
            'intervention_kendall_p': intervention_kendall_p,
            'kendall_improvement': kendall_improvement,
            # Advanced baseline metrics
            'baseline_entropy': baseline_advanced['entropy'],
            'baseline_gini_diversity': baseline_advanced['gini_diversity'],
            'baseline_js_divergence': baseline_advanced['js_divergence'],
            'baseline_total_variation': baseline_advanced['total_variation'],
            'baseline_macro_f1': baseline_advanced['macro_f1'],
            'baseline_balanced_accuracy': baseline_advanced['balanced_accuracy'],
            'baseline_cohen_kappa': baseline_advanced['cohen_kappa'],
            'baseline_dist_quality_score': baseline_advanced['dist_quality_score'],
            # Advanced intervention metrics
            'intervention_entropy': intervention_advanced['entropy'],
            'intervention_gini_diversity': intervention_advanced['gini_diversity'],
            'intervention_js_divergence': intervention_advanced['js_divergence'],
            'intervention_total_variation': intervention_advanced['total_variation'],
            'intervention_macro_f1': intervention_advanced['macro_f1'],
            'intervention_balanced_accuracy': intervention_advanced['balanced_accuracy'],
            'intervention_cohen_kappa': intervention_advanced['cohen_kappa'],
            'intervention_dist_quality_score': intervention_advanced['dist_quality_score'],
            # Improvements
            **improvements,
            'n_samples': len(test_users),
            # Intersectional-specific metadata
            'demographics_combined': demographic_attrs,
            'n_unique_combinations': unique_combos,
            'combination_distribution': dict(combo_dist)
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

    # INTERSECTIONAL MODE: Combine multiple demographics
    if args.intersect_demographics is not None:
        print(f"\n{'='*80}")
        print(f"INTERSECTIONAL INTERVENTION MODE")
        print(f"Combining demographics: {', '.join(args.intersect_demographics)}")
        print(f"{'='*80}")

        # Load extraction files for all specified demographics
        demographic_fold_files = {}
        demographic_categories = {}

        for demographic in args.intersect_demographics:
            print(f"\nLoading extractions for {demographic}...")
            fold_files = []
            for fold_idx in range(args.n_folds):
                fold_file = extraction_dir / f"{demographic}_{args.probe_type}_{args.run_id}_fold{fold_idx + 1}_extractions.pkl"
                if not fold_file.exists():
                    raise FileNotFoundError(
                        f"Missing extraction file: {fold_file}\n"
                        f"Please run extraction phase for {demographic} first with the same run_id"
                    )
                fold_files.append(fold_file)
            demographic_fold_files[demographic] = fold_files
            print(f"  Found {len(fold_files)} fold files for {demographic}")

            # Load first fold to get category names
            with open(fold_files[0], 'rb') as f:
                sample_data = pickle.load(f)
            question_key = list(sample_data['question_extractions'].keys())[0]
            demographic_categories[demographic] = sample_data['question_extractions'][question_key]['category_names']
            print(f"  Categories: {demographic_categories[demographic]}")

        # Process intersectional folds
        fold_results = []

        for fold_idx in range(args.n_folds):
            print(f"\n{'-'*80}")
            print(f"FOLD {fold_idx + 1}/{args.n_folds}")
            print(f"{'-'*80}")

            # Load all demographic extractions for this fold
            demographic_weights_dict = {}
            test_questions = None
            train_questions = None

            for demographic in args.intersect_demographics:
                fold_file = demographic_fold_files[demographic][fold_idx]
                with open(fold_file, 'rb') as f:
                    extraction_data = pickle.load(f)

                # Validate fold consistency
                if test_questions is None:
                    test_questions = extraction_data['test_questions']
                    train_questions = extraction_data['train_questions']
                else:
                    if extraction_data['test_questions'] != test_questions:
                        raise ValueError(
                            f"Fold {fold_idx+1}: Test questions don't match between demographics!\n"
                            f"This likely means extractions were run with different random seeds or folds."
                        )

                # Get intervention weights from probing results and select top-k per demographic
                all_weights = extraction_data['intervention_weights']

                # Sort by coefficient magnitude and select top-k for THIS demographic
                sorted_demo_weights = dict(sorted(
                    all_weights.items(),
                    key=lambda item: np.linalg.norm(item[1][0]),  # Sort by coefficient magnitude
                    reverse=True
                ))

                # Keep only top-k components from this demographic
                top_k_demo_weights = dict(list(sorted_demo_weights.items())[:args.top_k_heads])
                demographic_weights_dict[demographic] = top_k_demo_weights

                print(f"  Loaded {demographic}: selected top {len(top_k_demo_weights)} of {len(all_weights)} components")

            print(f"\nTest questions ({len(test_questions)}): {test_questions}")

            # Evaluate intersectional intervention on test fold
            print(f"\nEvaluating intersectional intervention on {len(test_questions)} test questions...")
            print(f"  Using top {args.top_k_heads} components from EACH demographic")
            test_results = evaluate_intersectional_intervention_on_fold(
                model, tokenizer, df, test_questions,
                args.intersect_demographics,
                demographic_weights_dict,
                demographic_categories,
                args.device,
                args.probe_type,
                args.intervention_strength,
                args.eval_sample_size,
                args.top_k_heads,
                args.prompt_style,
                results_dir,
                args.run_id
            )

            # Aggregate fold results (same as single-demographic case)
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

                # Calculate advanced metrics averages
                def safe_mean(metric_name):
                    """Calculate mean for a metric, filtering out None values"""
                    values = [r[metric_name] for r in test_results.values()
                             if r.get(metric_name) is not None and not (isinstance(r.get(metric_name), float) and np.isnan(r.get(metric_name)))]
                    return np.mean(values) if values else None

                # Key advanced metrics
                fold_baseline_entropy = safe_mean('baseline_entropy')
                fold_intervention_entropy = safe_mean('intervention_entropy')
                fold_baseline_gini = safe_mean('baseline_gini_diversity')
                fold_intervention_gini = safe_mean('intervention_gini_diversity')
                fold_baseline_js_div = safe_mean('baseline_js_divergence')
                fold_intervention_js_div = safe_mean('intervention_js_divergence')
                fold_baseline_total_var = safe_mean('baseline_total_variation')
                fold_intervention_total_var = safe_mean('intervention_total_variation')
                fold_baseline_macro_f1 = safe_mean('baseline_macro_f1')
                fold_intervention_macro_f1 = safe_mean('intervention_macro_f1')
                fold_baseline_balanced_acc = safe_mean('baseline_balanced_accuracy')
                fold_intervention_balanced_acc = safe_mean('intervention_balanced_accuracy')
                fold_baseline_cohen = safe_mean('baseline_cohen_kappa')
                fold_intervention_cohen = safe_mean('intervention_cohen_kappa')
                fold_baseline_dist_quality = safe_mean('baseline_dist_quality_score')
                fold_intervention_dist_quality = safe_mean('intervention_dist_quality_score')

                # Improvements
                fold_entropy_improvement = safe_mean('entropy_improvement')
                fold_gini_improvement = safe_mean('gini_diversity_improvement')
                fold_js_div_improvement = safe_mean('js_divergence_improvement')
                fold_total_var_improvement = safe_mean('total_variation_improvement')
                fold_macro_f1_improvement = safe_mean('macro_f1_improvement')
                fold_balanced_acc_improvement = safe_mean('balanced_accuracy_improvement')
                fold_cohen_improvement = safe_mean('cohen_kappa_improvement')
                fold_dist_quality_improvement = safe_mean('dist_quality_score_improvement')

                print(f"\nFold {fold_idx + 1} Results:")
                print(f"  Baseline:     {fold_baseline*100:.1f}%")
                print(f"  Intervention: {fold_intervention*100:.1f}%")
                print(f"  Improvement:  {fold_improvement:+.1f} points")

                # Print Kendall's tau
                if fold_baseline_kendall is not None or fold_intervention_kendall is not None:
                    baseline_str = f"{fold_baseline_kendall:.3f}" if fold_baseline_kendall is not None else "N/A"
                    intervention_str = f"{fold_intervention_kendall:.3f}" if fold_intervention_kendall is not None else "N/A"
                    improvement_str = f"{fold_kendall_improvement:+.3f}" if fold_kendall_improvement is not None else "N/A"
                    print(f"  Kendall's tau (Baseline):     {baseline_str}")
                    print(f"  Kendall's tau (Intervention): {intervention_str}")
                    print(f"  Kendall's tau Improvement:    {improvement_str}")

                # Print key advanced metrics
                print(f"\n  Advanced Metrics:")
                if fold_baseline_entropy is not None:
                    print(f"    Entropy (Baseline → Intervention):     {fold_baseline_entropy:.3f} → {fold_intervention_entropy:.3f} ({fold_entropy_improvement:+.3f})")
                if fold_baseline_js_div is not None:
                    print(f"    JS Divergence (Baseline → Intervention): {fold_baseline_js_div:.3f} → {fold_intervention_js_div:.3f} ({fold_js_div_improvement:+.3f})")
                if fold_baseline_macro_f1 is not None:
                    print(f"    Macro F1 (Baseline → Intervention):    {fold_baseline_macro_f1:.3f} → {fold_intervention_macro_f1:.3f} ({fold_macro_f1_improvement:+.3f})")
                if fold_baseline_dist_quality is not None:
                    print(f"    Dist Quality (Baseline → Intervention): {fold_baseline_dist_quality:.3f} → {fold_intervention_dist_quality:.3f} ({fold_dist_quality_improvement:+.3f})")

                fold_aggregate_metrics = {
                    'baseline_accuracy': fold_baseline,
                    'intervention_accuracy': fold_intervention,
                    'improvement': fold_improvement,
                    'baseline_kendall_tau': fold_baseline_kendall,
                    'intervention_kendall_tau': fold_intervention_kendall,
                    'kendall_improvement': fold_kendall_improvement,
                    # Advanced metrics - Diversity
                    'baseline_entropy': fold_baseline_entropy,
                    'intervention_entropy': fold_intervention_entropy,
                    'entropy_improvement': fold_entropy_improvement,
                    'baseline_gini_diversity': fold_baseline_gini,
                    'intervention_gini_diversity': fold_intervention_gini,
                    'gini_diversity_improvement': fold_gini_improvement,
                    # Advanced metrics - Distributional Similarity
                    'baseline_js_divergence': fold_baseline_js_div,
                    'intervention_js_divergence': fold_intervention_js_div,
                    'js_divergence_improvement': fold_js_div_improvement,
                    'baseline_total_variation': fold_baseline_total_var,
                    'intervention_total_variation': fold_intervention_total_var,
                    'total_variation_improvement': fold_total_var_improvement,
                    # Advanced metrics - Multiclass Performance
                    'baseline_macro_f1': fold_baseline_macro_f1,
                    'intervention_macro_f1': fold_intervention_macro_f1,
                    'macro_f1_improvement': fold_macro_f1_improvement,
                    'baseline_balanced_accuracy': fold_baseline_balanced_acc,
                    'intervention_balanced_accuracy': fold_intervention_balanced_acc,
                    'balanced_accuracy_improvement': fold_balanced_acc_improvement,
                    'baseline_cohen_kappa': fold_baseline_cohen,
                    'intervention_cohen_kappa': fold_intervention_cohen,
                    'cohen_kappa_improvement': fold_cohen_improvement,
                    # Combined Quality Score
                    'baseline_dist_quality_score': fold_baseline_dist_quality,
                    'intervention_dist_quality_score': fold_intervention_dist_quality,
                    'dist_quality_score_improvement': fold_dist_quality_improvement,
                    'n_test_questions': len(test_results)
                }
            else:
                fold_aggregate_metrics = None

            fold_results.append({
                'fold': fold_idx,
                'test_questions': test_questions,
                'test_results': test_results,
                'aggregate_metrics': fold_aggregate_metrics
            })

        # Aggregate results across folds
        print(f"\n{'='*80}")
        print(f"AGGREGATING INTERSECTIONAL RESULTS")
        print(f"{'='*80}")

        # Calculate overall metrics
        fold_baseline_accs = [fd['aggregate_metrics']['baseline_accuracy'] for fd in fold_results if fd['aggregate_metrics']]
        fold_intervention_accs = [fd['aggregate_metrics']['intervention_accuracy'] for fd in fold_results if fd['aggregate_metrics']]
        fold_improvements = [fd['aggregate_metrics']['improvement'] for fd in fold_results if fd['aggregate_metrics']]

        n_folds = len(fold_baseline_accs)
        overall_baseline_mean = np.mean(fold_baseline_accs)
        overall_baseline_std = np.std(fold_baseline_accs, ddof=1) if n_folds > 1 else None
        overall_intervention_mean = np.mean(fold_intervention_accs)
        overall_intervention_std = np.std(fold_intervention_accs, ddof=1) if n_folds > 1 else None
        overall_improvement_mean = np.mean(fold_improvements)
        overall_improvement_std = np.std(fold_improvements, ddof=1) if n_folds > 1 else None

        # Aggregate advanced metrics across folds
        def aggregate_metric(metric_name):
            """Extract and aggregate a metric across folds (mean and std)"""
            values = [fd['aggregate_metrics'][metric_name] for fd in fold_results
                     if fd['aggregate_metrics'] and fd['aggregate_metrics'].get(metric_name) is not None
                     and not (isinstance(fd['aggregate_metrics'].get(metric_name), float) and np.isnan(fd['aggregate_metrics'].get(metric_name)))]
            mean = np.mean(values) if values else None
            std = np.std(values, ddof=1) if len(values) > 1 else None
            return mean, std

        # Kendall's tau
        overall_baseline_kendall_mean, overall_baseline_kendall_std = aggregate_metric('baseline_kendall_tau')
        overall_intervention_kendall_mean, overall_intervention_kendall_std = aggregate_metric('intervention_kendall_tau')
        overall_kendall_improvement_mean, overall_kendall_improvement_std = aggregate_metric('kendall_improvement')

        # Advanced metrics - Diversity
        overall_baseline_entropy_mean, overall_baseline_entropy_std = aggregate_metric('baseline_entropy')
        overall_intervention_entropy_mean, overall_intervention_entropy_std = aggregate_metric('intervention_entropy')
        overall_entropy_improvement_mean, overall_entropy_improvement_std = aggregate_metric('entropy_improvement')
        overall_baseline_gini_mean, overall_baseline_gini_std = aggregate_metric('baseline_gini_diversity')
        overall_intervention_gini_mean, overall_intervention_gini_std = aggregate_metric('intervention_gini_diversity')
        overall_gini_improvement_mean, overall_gini_improvement_std = aggregate_metric('gini_diversity_improvement')

        # Advanced metrics - Distributional Similarity
        overall_baseline_js_div_mean, overall_baseline_js_div_std = aggregate_metric('baseline_js_divergence')
        overall_intervention_js_div_mean, overall_intervention_js_div_std = aggregate_metric('intervention_js_divergence')
        overall_js_div_improvement_mean, overall_js_div_improvement_std = aggregate_metric('js_divergence_improvement')
        overall_baseline_total_var_mean, overall_baseline_total_var_std = aggregate_metric('baseline_total_variation')
        overall_intervention_total_var_mean, overall_intervention_total_var_std = aggregate_metric('intervention_total_variation')
        overall_total_var_improvement_mean, overall_total_var_improvement_std = aggregate_metric('total_variation_improvement')

        # Advanced metrics - Multiclass Performance
        overall_baseline_macro_f1_mean, overall_baseline_macro_f1_std = aggregate_metric('baseline_macro_f1')
        overall_intervention_macro_f1_mean, overall_intervention_macro_f1_std = aggregate_metric('intervention_macro_f1')
        overall_macro_f1_improvement_mean, overall_macro_f1_improvement_std = aggregate_metric('macro_f1_improvement')
        overall_baseline_balanced_acc_mean, overall_baseline_balanced_acc_std = aggregate_metric('baseline_balanced_accuracy')
        overall_intervention_balanced_acc_mean, overall_intervention_balanced_acc_std = aggregate_metric('intervention_balanced_accuracy')
        overall_balanced_acc_improvement_mean, overall_balanced_acc_improvement_std = aggregate_metric('balanced_accuracy_improvement')
        overall_baseline_cohen_mean, overall_baseline_cohen_std = aggregate_metric('baseline_cohen_kappa')
        overall_intervention_cohen_mean, overall_intervention_cohen_std = aggregate_metric('intervention_cohen_kappa')
        overall_cohen_improvement_mean, overall_cohen_improvement_std = aggregate_metric('cohen_kappa_improvement')

        # Combined Quality Score
        overall_baseline_dist_quality_mean, overall_baseline_dist_quality_std = aggregate_metric('baseline_dist_quality_score')
        overall_intervention_dist_quality_mean, overall_intervention_dist_quality_std = aggregate_metric('intervention_dist_quality_score')
        overall_dist_quality_improvement_mean, overall_dist_quality_improvement_std = aggregate_metric('dist_quality_score_improvement')

        print(f"\nOverall Results (mean ± std across {n_folds} folds):")
        if overall_baseline_std is not None:
            print(f"  Baseline:     {overall_baseline_mean*100:.1f}% ± {overall_baseline_std*100:.1f}%")
            print(f"  Intervention: {overall_intervention_mean*100:.1f}% ± {overall_intervention_std*100:.1f}%")
            print(f"  Improvement:  {overall_improvement_mean:+.1f} ± {overall_improvement_std:.1f} points")
        else:
            print(f"  Baseline:     {overall_baseline_mean*100:.1f}%")
            print(f"  Intervention: {overall_intervention_mean*100:.1f}%")
            print(f"  Improvement:  {overall_improvement_mean:+.1f} points")

        # Save intersectional results
        intersect_name = "+".join(args.intersect_demographics)
        intersectional_results = {
            'mode': 'intersectional',
            'demographics_combined': args.intersect_demographics,
            'demographic_categories': demographic_categories,
            'probe_type': args.probe_type,
            'run_id': args.run_id,
            'n_folds': args.n_folds,
            'top_k_heads': args.top_k_heads,
            'intervention_strength': args.intervention_strength,
            'fold_results': fold_results,
            'overall_metrics': {
                # Basic accuracy metrics
                'baseline_accuracy_mean': overall_baseline_mean,
                'baseline_accuracy_std': overall_baseline_std,
                'intervention_accuracy_mean': overall_intervention_mean,
                'intervention_accuracy_std': overall_intervention_std,
                'improvement_mean': overall_improvement_mean,
                'improvement_std': overall_improvement_std,
                # Kendall's tau (ordinal correlation)
                'baseline_kendall_tau_mean': overall_baseline_kendall_mean,
                'baseline_kendall_tau_std': overall_baseline_kendall_std,
                'intervention_kendall_tau_mean': overall_intervention_kendall_mean,
                'intervention_kendall_tau_std': overall_intervention_kendall_std,
                'kendall_improvement_mean': overall_kendall_improvement_mean,
                'kendall_improvement_std': overall_kendall_improvement_std,
                # Advanced metrics - Diversity
                'baseline_entropy_mean': overall_baseline_entropy_mean,
                'baseline_entropy_std': overall_baseline_entropy_std,
                'intervention_entropy_mean': overall_intervention_entropy_mean,
                'intervention_entropy_std': overall_intervention_entropy_std,
                'entropy_improvement_mean': overall_entropy_improvement_mean,
                'entropy_improvement_std': overall_entropy_improvement_std,
                'baseline_gini_diversity_mean': overall_baseline_gini_mean,
                'baseline_gini_diversity_std': overall_baseline_gini_std,
                'intervention_gini_diversity_mean': overall_intervention_gini_mean,
                'intervention_gini_diversity_std': overall_intervention_gini_std,
                'gini_diversity_improvement_mean': overall_gini_improvement_mean,
                'gini_diversity_improvement_std': overall_gini_improvement_std,
                # Advanced metrics - Distributional Similarity
                'baseline_js_divergence_mean': overall_baseline_js_div_mean,
                'baseline_js_divergence_std': overall_baseline_js_div_std,
                'intervention_js_divergence_mean': overall_intervention_js_div_mean,
                'intervention_js_divergence_std': overall_intervention_js_div_std,
                'js_divergence_improvement_mean': overall_js_div_improvement_mean,
                'js_divergence_improvement_std': overall_js_div_improvement_std,
                'baseline_total_variation_mean': overall_baseline_total_var_mean,
                'baseline_total_variation_std': overall_baseline_total_var_std,
                'intervention_total_variation_mean': overall_intervention_total_var_mean,
                'intervention_total_variation_std': overall_intervention_total_var_std,
                'total_variation_improvement_mean': overall_total_var_improvement_mean,
                'total_variation_improvement_std': overall_total_var_improvement_std,
                # Advanced metrics - Multiclass Performance
                'baseline_macro_f1_mean': overall_baseline_macro_f1_mean,
                'baseline_macro_f1_std': overall_baseline_macro_f1_std,
                'intervention_macro_f1_mean': overall_intervention_macro_f1_mean,
                'intervention_macro_f1_std': overall_intervention_macro_f1_std,
                'macro_f1_improvement_mean': overall_macro_f1_improvement_mean,
                'macro_f1_improvement_std': overall_macro_f1_improvement_std,
                'baseline_balanced_accuracy_mean': overall_baseline_balanced_acc_mean,
                'baseline_balanced_accuracy_std': overall_baseline_balanced_acc_std,
                'intervention_balanced_accuracy_mean': overall_intervention_balanced_acc_mean,
                'intervention_balanced_accuracy_std': overall_intervention_balanced_acc_std,
                'balanced_accuracy_improvement_mean': overall_balanced_acc_improvement_mean,
                'balanced_accuracy_improvement_std': overall_balanced_acc_improvement_std,
                'baseline_cohen_kappa_mean': overall_baseline_cohen_mean,
                'baseline_cohen_kappa_std': overall_baseline_cohen_std,
                'intervention_cohen_kappa_mean': overall_intervention_cohen_mean,
                'intervention_cohen_kappa_std': overall_intervention_cohen_std,
                'cohen_kappa_improvement_mean': overall_cohen_improvement_mean,
                'cohen_kappa_improvement_std': overall_cohen_improvement_std,
                # Combined Quality Score
                'baseline_dist_quality_score_mean': overall_baseline_dist_quality_mean,
                'baseline_dist_quality_score_std': overall_baseline_dist_quality_std,
                'intervention_dist_quality_score_mean': overall_intervention_dist_quality_mean,
                'intervention_dist_quality_score_std': overall_intervention_dist_quality_std,
                'dist_quality_score_improvement_mean': overall_dist_quality_improvement_mean,
                'dist_quality_score_improvement_std': overall_dist_quality_improvement_std,
            },
            'timestamp': datetime.now().isoformat()
        }

        result_file = results_dir / f"intersectional_{intersect_name}_{args.probe_type}_k{args.top_k_heads}_s{args.intervention_strength}_{args.run_id}_intervention_results.pkl"
        with open(result_file, 'wb') as f:
            pickle.dump(intersectional_results, f)
        print(f"\nSaved intersectional results: {result_file}")

        all_demographic_results[f'intersectional_{intersect_name}'] = intersectional_results

        # Skip normal demographic loop
        print(f"\n{'='*80}")
        print("INTERSECTIONAL INTERVENTION PHASE COMPLETE")
        print(f"{'='*80}")

        # Save summary
        summary_file = results_dir / f'intervention_summary_{args.probe_type}_k{args.top_k_heads}_s{args.intervention_strength}_{args.run_id}.json'
        summary_data = {
            'config': vars(args),
            'mode': 'intersectional',
            'results': {
                f'intersectional_{intersect_name}': {
                    'demographics_combined': args.intersect_demographics,
                    # Basic accuracy metrics
                    'baseline_accuracy_mean': intersectional_results['overall_metrics']['baseline_accuracy_mean'],
                    'baseline_accuracy_std': intersectional_results['overall_metrics']['baseline_accuracy_std'],
                    'intervention_accuracy_mean': intersectional_results['overall_metrics']['intervention_accuracy_mean'],
                    'intervention_accuracy_std': intersectional_results['overall_metrics']['intervention_accuracy_std'],
                    'improvement_mean': intersectional_results['overall_metrics']['improvement_mean'],
                    'improvement_std': intersectional_results['overall_metrics']['improvement_std'],

                    # Kendall's tau (ordinal correlation)
                    'baseline_kendall_tau_mean': intersectional_results['overall_metrics']['baseline_kendall_tau_mean'],
                    'baseline_kendall_tau_std': intersectional_results['overall_metrics']['baseline_kendall_tau_std'],
                    'intervention_kendall_tau_mean': intersectional_results['overall_metrics']['intervention_kendall_tau_mean'],
                    'intervention_kendall_tau_std': intersectional_results['overall_metrics']['intervention_kendall_tau_std'],
                    'kendall_improvement_mean': intersectional_results['overall_metrics']['kendall_improvement_mean'],
                    'kendall_improvement_std': intersectional_results['overall_metrics']['kendall_improvement_std'],

                    # Advanced metrics - Diversity
                    'baseline_entropy_mean': intersectional_results['overall_metrics']['baseline_entropy_mean'],
                    'baseline_entropy_std': intersectional_results['overall_metrics']['baseline_entropy_std'],
                    'intervention_entropy_mean': intersectional_results['overall_metrics']['intervention_entropy_mean'],
                    'intervention_entropy_std': intersectional_results['overall_metrics']['intervention_entropy_std'],
                    'entropy_improvement_mean': intersectional_results['overall_metrics']['entropy_improvement_mean'],
                    'entropy_improvement_std': intersectional_results['overall_metrics']['entropy_improvement_std'],

                    'baseline_gini_diversity_mean': intersectional_results['overall_metrics']['baseline_gini_diversity_mean'],
                    'baseline_gini_diversity_std': intersectional_results['overall_metrics']['baseline_gini_diversity_std'],
                    'intervention_gini_diversity_mean': intersectional_results['overall_metrics']['intervention_gini_diversity_mean'],
                    'intervention_gini_diversity_std': intersectional_results['overall_metrics']['intervention_gini_diversity_std'],
                    'gini_diversity_improvement_mean': intersectional_results['overall_metrics']['gini_diversity_improvement_mean'],
                    'gini_diversity_improvement_std': intersectional_results['overall_metrics']['gini_diversity_improvement_std'],

                    # Advanced metrics - Distributional Similarity
                    'baseline_js_divergence_mean': intersectional_results['overall_metrics']['baseline_js_divergence_mean'],
                    'baseline_js_divergence_std': intersectional_results['overall_metrics']['baseline_js_divergence_std'],
                    'intervention_js_divergence_mean': intersectional_results['overall_metrics']['intervention_js_divergence_mean'],
                    'intervention_js_divergence_std': intersectional_results['overall_metrics']['intervention_js_divergence_std'],
                    'js_divergence_improvement_mean': intersectional_results['overall_metrics']['js_divergence_improvement_mean'],
                    'js_divergence_improvement_std': intersectional_results['overall_metrics']['js_divergence_improvement_std'],

                    'baseline_total_variation_mean': intersectional_results['overall_metrics']['baseline_total_variation_mean'],
                    'baseline_total_variation_std': intersectional_results['overall_metrics']['baseline_total_variation_std'],
                    'intervention_total_variation_mean': intersectional_results['overall_metrics']['intervention_total_variation_mean'],
                    'intervention_total_variation_std': intersectional_results['overall_metrics']['intervention_total_variation_std'],
                    'total_variation_improvement_mean': intersectional_results['overall_metrics']['total_variation_improvement_mean'],
                    'total_variation_improvement_std': intersectional_results['overall_metrics']['total_variation_improvement_std'],

                    # Advanced metrics - Multiclass Performance
                    'baseline_macro_f1_mean': intersectional_results['overall_metrics']['baseline_macro_f1_mean'],
                    'baseline_macro_f1_std': intersectional_results['overall_metrics']['baseline_macro_f1_std'],
                    'intervention_macro_f1_mean': intersectional_results['overall_metrics']['intervention_macro_f1_mean'],
                    'intervention_macro_f1_std': intersectional_results['overall_metrics']['intervention_macro_f1_std'],
                    'macro_f1_improvement_mean': intersectional_results['overall_metrics']['macro_f1_improvement_mean'],
                    'macro_f1_improvement_std': intersectional_results['overall_metrics']['macro_f1_improvement_std'],

                    'baseline_balanced_accuracy_mean': intersectional_results['overall_metrics']['baseline_balanced_accuracy_mean'],
                    'baseline_balanced_accuracy_std': intersectional_results['overall_metrics']['baseline_balanced_accuracy_std'],
                    'intervention_balanced_accuracy_mean': intersectional_results['overall_metrics']['intervention_balanced_accuracy_mean'],
                    'intervention_balanced_accuracy_std': intersectional_results['overall_metrics']['intervention_balanced_accuracy_std'],
                    'balanced_accuracy_improvement_mean': intersectional_results['overall_metrics']['balanced_accuracy_improvement_mean'],
                    'balanced_accuracy_improvement_std': intersectional_results['overall_metrics']['balanced_accuracy_improvement_std'],

                    'baseline_cohen_kappa_mean': intersectional_results['overall_metrics']['baseline_cohen_kappa_mean'],
                    'baseline_cohen_kappa_std': intersectional_results['overall_metrics']['baseline_cohen_kappa_std'],
                    'intervention_cohen_kappa_mean': intersectional_results['overall_metrics']['intervention_cohen_kappa_mean'],
                    'intervention_cohen_kappa_std': intersectional_results['overall_metrics']['intervention_cohen_kappa_std'],
                    'cohen_kappa_improvement_mean': intersectional_results['overall_metrics']['cohen_kappa_improvement_mean'],
                    'cohen_kappa_improvement_std': intersectional_results['overall_metrics']['cohen_kappa_improvement_std'],

                    # Combined Quality Score
                    'baseline_dist_quality_score_mean': intersectional_results['overall_metrics']['baseline_dist_quality_score_mean'],
                    'baseline_dist_quality_score_std': intersectional_results['overall_metrics']['baseline_dist_quality_score_std'],
                    'intervention_dist_quality_score_mean': intersectional_results['overall_metrics']['intervention_dist_quality_score_mean'],
                    'intervention_dist_quality_score_std': intersectional_results['overall_metrics']['intervention_dist_quality_score_std'],
                    'dist_quality_score_improvement_mean': intersectional_results['overall_metrics']['dist_quality_score_improvement_mean'],
                    'dist_quality_score_improvement_std': intersectional_results['overall_metrics']['dist_quality_score_improvement_std'],
                }
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"\nSummary saved: {summary_file}")

        return  # Exit early for intersectional mode

    # NORMAL MODE: Process each demographic individually
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
                args.probe_type, args.intervention_strength, args.eval_sample_size,
                args.prompt_style,
                results_dir,
                args.run_id,
                args.top_k_heads
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

                # Calculate advanced metrics averages
                def safe_mean(metric_name):
                    """Calculate mean for a metric, filtering out None values"""
                    values = [r[metric_name] for r in test_results.values()
                             if r.get(metric_name) is not None and not (isinstance(r.get(metric_name), float) and np.isnan(r.get(metric_name)))]
                    return np.mean(values) if values else None

                # Key advanced metrics
                fold_baseline_entropy = safe_mean('baseline_entropy')
                fold_intervention_entropy = safe_mean('intervention_entropy')
                fold_baseline_gini = safe_mean('baseline_gini_diversity')
                fold_intervention_gini = safe_mean('intervention_gini_diversity')
                fold_baseline_js_div = safe_mean('baseline_js_divergence')
                fold_intervention_js_div = safe_mean('intervention_js_divergence')
                fold_baseline_total_var = safe_mean('baseline_total_variation')
                fold_intervention_total_var = safe_mean('intervention_total_variation')
                fold_baseline_macro_f1 = safe_mean('baseline_macro_f1')
                fold_intervention_macro_f1 = safe_mean('intervention_macro_f1')
                fold_baseline_balanced_acc = safe_mean('baseline_balanced_accuracy')
                fold_intervention_balanced_acc = safe_mean('intervention_balanced_accuracy')
                fold_baseline_cohen = safe_mean('baseline_cohen_kappa')
                fold_intervention_cohen = safe_mean('intervention_cohen_kappa')
                fold_baseline_dist_quality = safe_mean('baseline_dist_quality_score')
                fold_intervention_dist_quality = safe_mean('intervention_dist_quality_score')

                # Improvements
                fold_entropy_improvement = safe_mean('entropy_improvement')
                fold_gini_improvement = safe_mean('gini_diversity_improvement')
                fold_js_div_improvement = safe_mean('js_divergence_improvement')
                fold_total_var_improvement = safe_mean('total_variation_improvement')
                fold_macro_f1_improvement = safe_mean('macro_f1_improvement')
                fold_balanced_acc_improvement = safe_mean('balanced_accuracy_improvement')
                fold_cohen_improvement = safe_mean('cohen_kappa_improvement')
                fold_dist_quality_improvement = safe_mean('dist_quality_score_improvement')

                print(f"\nFold {fold_idx + 1} Results:")
                print(f"  Baseline:     {fold_baseline*100:.1f}%")
                print(f"  Intervention: {fold_intervention*100:.1f}%")
                print(f"  Improvement:  {fold_improvement:+.1f} points")

                # Print Kendall's tau
                if fold_baseline_kendall is not None or fold_intervention_kendall is not None:
                    baseline_str = f"{fold_baseline_kendall:.3f}" if fold_baseline_kendall is not None else "N/A"
                    intervention_str = f"{fold_intervention_kendall:.3f}" if fold_intervention_kendall is not None else "N/A"
                    improvement_str = f"{fold_kendall_improvement:+.3f}" if fold_kendall_improvement is not None else "N/A"
                    print(f"  Kendall's tau (Baseline):     {baseline_str}")
                    print(f"  Kendall's tau (Intervention): {intervention_str}")
                    print(f"  Kendall's tau Improvement:    {improvement_str}")

                # Print key advanced metrics
                print(f"\n  Advanced Metrics:")
                if fold_baseline_entropy is not None:
                    print(f"    Entropy (Baseline → Intervention):     {fold_baseline_entropy:.3f} → {fold_intervention_entropy:.3f} ({fold_entropy_improvement:+.3f})")
                if fold_baseline_js_div is not None:
                    print(f"    JS Divergence (Baseline → Intervention): {fold_baseline_js_div:.3f} → {fold_intervention_js_div:.3f} ({fold_js_div_improvement:+.3f})")
                if fold_baseline_macro_f1 is not None:
                    print(f"    Macro F1 (Baseline → Intervention):    {fold_baseline_macro_f1:.3f} → {fold_intervention_macro_f1:.3f} ({fold_macro_f1_improvement:+.3f})")
                if fold_baseline_dist_quality is not None:
                    print(f"    Dist Quality (Baseline → Intervention): {fold_baseline_dist_quality:.3f} → {fold_intervention_dist_quality:.3f} ({fold_dist_quality_improvement:+.3f})")

                # Store fold-level aggregates for cross-fold statistics
                fold_aggregate_metrics = {
                    'baseline_accuracy': fold_baseline,
                    'intervention_accuracy': fold_intervention,
                    'improvement': fold_improvement,
                    'baseline_kendall_tau': fold_baseline_kendall,
                    'intervention_kendall_tau': fold_intervention_kendall,
                    'kendall_improvement': fold_kendall_improvement,
                    # Advanced metrics - Diversity
                    'baseline_entropy': fold_baseline_entropy,
                    'intervention_entropy': fold_intervention_entropy,
                    'entropy_improvement': fold_entropy_improvement,
                    'baseline_gini_diversity': fold_baseline_gini,
                    'intervention_gini_diversity': fold_intervention_gini,
                    'gini_diversity_improvement': fold_gini_improvement,
                    # Advanced metrics - Distributional Similarity
                    'baseline_js_divergence': fold_baseline_js_div,
                    'intervention_js_divergence': fold_intervention_js_div,
                    'js_divergence_improvement': fold_js_div_improvement,
                    'baseline_total_variation': fold_baseline_total_var,
                    'intervention_total_variation': fold_intervention_total_var,
                    'total_variation_improvement': fold_total_var_improvement,
                    # Advanced metrics - Multiclass Performance
                    'baseline_macro_f1': fold_baseline_macro_f1,
                    'intervention_macro_f1': fold_intervention_macro_f1,
                    'macro_f1_improvement': fold_macro_f1_improvement,
                    'baseline_balanced_accuracy': fold_baseline_balanced_acc,
                    'intervention_balanced_accuracy': fold_intervention_balanced_acc,
                    'balanced_accuracy_improvement': fold_balanced_acc_improvement,
                    'baseline_cohen_kappa': fold_baseline_cohen,
                    'intervention_cohen_kappa': fold_intervention_cohen,
                    'cohen_kappa_improvement': fold_cohen_improvement,
                    # Combined Quality Score
                    'baseline_dist_quality_score': fold_baseline_dist_quality,
                    'intervention_dist_quality_score': fold_intervention_dist_quality,
                    'dist_quality_score_improvement': fold_dist_quality_improvement,
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

        # Advanced metrics lists
        fold_baseline_entropies = []
        fold_intervention_entropies = []
        fold_entropy_improvements = []
        fold_baseline_ginis = []
        fold_intervention_ginis = []
        fold_gini_improvements = []
        fold_baseline_js_divs = []
        fold_intervention_js_divs = []
        fold_js_div_improvements = []
        fold_baseline_total_vars = []
        fold_intervention_total_vars = []
        fold_total_var_improvements = []
        fold_baseline_macro_f1s = []
        fold_intervention_macro_f1s = []
        fold_macro_f1_improvements = []
        fold_baseline_balanced_accs = []
        fold_intervention_balanced_accs = []
        fold_balanced_acc_improvements = []
        fold_baseline_cohens = []
        fold_intervention_cohens = []
        fold_cohen_improvements = []
        fold_baseline_dist_qualities = []
        fold_intervention_dist_qualities = []
        fold_dist_quality_improvements = []

        for fold_data in fold_results:
            if fold_data['aggregate_metrics'] is not None:
                fold_baseline_accs.append(fold_data['aggregate_metrics']['baseline_accuracy'])
                fold_intervention_accs.append(fold_data['aggregate_metrics']['intervention_accuracy'])
                fold_improvements.append(fold_data['aggregate_metrics']['improvement'])

                # Add Kendall metrics if available (check each independently)
                if fold_data['aggregate_metrics']['baseline_kendall_tau'] is not None:
                    fold_baseline_kendalls.append(fold_data['aggregate_metrics']['baseline_kendall_tau'])
                if fold_data['aggregate_metrics']['intervention_kendall_tau'] is not None:
                    fold_intervention_kendalls.append(fold_data['aggregate_metrics']['intervention_kendall_tau'])
                if fold_data['aggregate_metrics']['kendall_improvement'] is not None:
                    fold_kendall_improvements.append(fold_data['aggregate_metrics']['kendall_improvement'])

                # Add advanced metrics if available
                # Entropy
                if fold_data['aggregate_metrics'].get('baseline_entropy') is not None:
                    fold_baseline_entropies.append(fold_data['aggregate_metrics']['baseline_entropy'])
                if fold_data['aggregate_metrics'].get('intervention_entropy') is not None:
                    fold_intervention_entropies.append(fold_data['aggregate_metrics']['intervention_entropy'])
                if fold_data['aggregate_metrics'].get('entropy_improvement') is not None:
                    fold_entropy_improvements.append(fold_data['aggregate_metrics']['entropy_improvement'])

                # Gini Diversity
                if fold_data['aggregate_metrics'].get('baseline_gini_diversity') is not None:
                    fold_baseline_ginis.append(fold_data['aggregate_metrics']['baseline_gini_diversity'])
                if fold_data['aggregate_metrics'].get('intervention_gini_diversity') is not None:
                    fold_intervention_ginis.append(fold_data['aggregate_metrics']['intervention_gini_diversity'])
                if fold_data['aggregate_metrics'].get('gini_diversity_improvement') is not None:
                    fold_gini_improvements.append(fold_data['aggregate_metrics']['gini_diversity_improvement'])

                # JS Divergence
                if fold_data['aggregate_metrics'].get('baseline_js_divergence') is not None:
                    fold_baseline_js_divs.append(fold_data['aggregate_metrics']['baseline_js_divergence'])
                if fold_data['aggregate_metrics'].get('intervention_js_divergence') is not None:
                    fold_intervention_js_divs.append(fold_data['aggregate_metrics']['intervention_js_divergence'])
                if fold_data['aggregate_metrics'].get('js_divergence_improvement') is not None:
                    fold_js_div_improvements.append(fold_data['aggregate_metrics']['js_divergence_improvement'])

                # Total Variation
                if fold_data['aggregate_metrics'].get('baseline_total_variation') is not None:
                    fold_baseline_total_vars.append(fold_data['aggregate_metrics']['baseline_total_variation'])
                if fold_data['aggregate_metrics'].get('intervention_total_variation') is not None:
                    fold_intervention_total_vars.append(fold_data['aggregate_metrics']['intervention_total_variation'])
                if fold_data['aggregate_metrics'].get('total_variation_improvement') is not None:
                    fold_total_var_improvements.append(fold_data['aggregate_metrics']['total_variation_improvement'])

                # Macro F1
                if fold_data['aggregate_metrics'].get('baseline_macro_f1') is not None:
                    fold_baseline_macro_f1s.append(fold_data['aggregate_metrics']['baseline_macro_f1'])
                if fold_data['aggregate_metrics'].get('intervention_macro_f1') is not None:
                    fold_intervention_macro_f1s.append(fold_data['aggregate_metrics']['intervention_macro_f1'])
                if fold_data['aggregate_metrics'].get('macro_f1_improvement') is not None:
                    fold_macro_f1_improvements.append(fold_data['aggregate_metrics']['macro_f1_improvement'])

                # Balanced Accuracy
                if fold_data['aggregate_metrics'].get('baseline_balanced_accuracy') is not None:
                    fold_baseline_balanced_accs.append(fold_data['aggregate_metrics']['baseline_balanced_accuracy'])
                if fold_data['aggregate_metrics'].get('intervention_balanced_accuracy') is not None:
                    fold_intervention_balanced_accs.append(fold_data['aggregate_metrics']['intervention_balanced_accuracy'])
                if fold_data['aggregate_metrics'].get('balanced_accuracy_improvement') is not None:
                    fold_balanced_acc_improvements.append(fold_data['aggregate_metrics']['balanced_accuracy_improvement'])

                # Cohen's Kappa
                if fold_data['aggregate_metrics'].get('baseline_cohen_kappa') is not None:
                    fold_baseline_cohens.append(fold_data['aggregate_metrics']['baseline_cohen_kappa'])
                if fold_data['aggregate_metrics'].get('intervention_cohen_kappa') is not None:
                    fold_intervention_cohens.append(fold_data['aggregate_metrics']['intervention_cohen_kappa'])
                if fold_data['aggregate_metrics'].get('cohen_kappa_improvement') is not None:
                    fold_cohen_improvements.append(fold_data['aggregate_metrics']['cohen_kappa_improvement'])

                # Distribution Quality Score
                if fold_data['aggregate_metrics'].get('baseline_dist_quality_score') is not None:
                    fold_baseline_dist_qualities.append(fold_data['aggregate_metrics']['baseline_dist_quality_score'])
                if fold_data['aggregate_metrics'].get('intervention_dist_quality_score') is not None:
                    fold_intervention_dist_qualities.append(fold_data['aggregate_metrics']['intervention_dist_quality_score'])
                if fold_data['aggregate_metrics'].get('dist_quality_score_improvement') is not None:
                    fold_dist_quality_improvements.append(fold_data['aggregate_metrics']['dist_quality_score_improvement'])

        n_folds = len(fold_baseline_accs)
        overall_baseline_mean = np.mean(fold_baseline_accs)
        overall_baseline_std = np.std(fold_baseline_accs, ddof=1) if n_folds > 1 else None

        overall_intervention_mean = np.mean(fold_intervention_accs)
        overall_intervention_std = np.std(fold_intervention_accs, ddof=1) if n_folds > 1 else None

        overall_improvement_mean = np.mean(fold_improvements)
        overall_improvement_std = np.std(fold_improvements, ddof=1) if n_folds > 1 else None

        # Kendall's tau statistics (calculate each independently as lists may have different lengths)
        if len(fold_baseline_kendalls) > 0:
            overall_baseline_kendall_mean = np.mean(fold_baseline_kendalls)
            overall_baseline_kendall_std = np.std(fold_baseline_kendalls, ddof=1) if len(fold_baseline_kendalls) > 1 else None
        else:
            overall_baseline_kendall_mean = overall_baseline_kendall_std = None

        if len(fold_intervention_kendalls) > 0:
            overall_intervention_kendall_mean = np.mean(fold_intervention_kendalls)
            overall_intervention_kendall_std = np.std(fold_intervention_kendalls, ddof=1) if len(fold_intervention_kendalls) > 1 else None
        else:
            overall_intervention_kendall_mean = overall_intervention_kendall_std = None

        if len(fold_kendall_improvements) > 0:
            overall_kendall_improvement_mean = np.mean(fold_kendall_improvements)
            overall_kendall_improvement_std = np.std(fold_kendall_improvements, ddof=1) if len(fold_kendall_improvements) > 1 else None
        else:
            overall_kendall_improvement_mean = overall_kendall_improvement_std = None

        # Advanced metrics statistics
        # Entropy
        if len(fold_baseline_entropies) > 0:
            overall_baseline_entropy_mean = np.mean(fold_baseline_entropies)
            overall_baseline_entropy_std = np.std(fold_baseline_entropies, ddof=1) if len(fold_baseline_entropies) > 1 else None
        else:
            overall_baseline_entropy_mean = overall_baseline_entropy_std = None

        if len(fold_intervention_entropies) > 0:
            overall_intervention_entropy_mean = np.mean(fold_intervention_entropies)
            overall_intervention_entropy_std = np.std(fold_intervention_entropies, ddof=1) if len(fold_intervention_entropies) > 1 else None
        else:
            overall_intervention_entropy_mean = overall_intervention_entropy_std = None

        if len(fold_entropy_improvements) > 0:
            overall_entropy_improvement_mean = np.mean(fold_entropy_improvements)
            overall_entropy_improvement_std = np.std(fold_entropy_improvements, ddof=1) if len(fold_entropy_improvements) > 1 else None
        else:
            overall_entropy_improvement_mean = overall_entropy_improvement_std = None

        # Gini Diversity
        if len(fold_baseline_ginis) > 0:
            overall_baseline_gini_mean = np.mean(fold_baseline_ginis)
            overall_baseline_gini_std = np.std(fold_baseline_ginis, ddof=1) if len(fold_baseline_ginis) > 1 else None
        else:
            overall_baseline_gini_mean = overall_baseline_gini_std = None

        if len(fold_intervention_ginis) > 0:
            overall_intervention_gini_mean = np.mean(fold_intervention_ginis)
            overall_intervention_gini_std = np.std(fold_intervention_ginis, ddof=1) if len(fold_intervention_ginis) > 1 else None
        else:
            overall_intervention_gini_mean = overall_intervention_gini_std = None

        if len(fold_gini_improvements) > 0:
            overall_gini_improvement_mean = np.mean(fold_gini_improvements)
            overall_gini_improvement_std = np.std(fold_gini_improvements, ddof=1) if len(fold_gini_improvements) > 1 else None
        else:
            overall_gini_improvement_mean = overall_gini_improvement_std = None

        # JS Divergence
        if len(fold_baseline_js_divs) > 0:
            overall_baseline_js_div_mean = np.mean(fold_baseline_js_divs)
            overall_baseline_js_div_std = np.std(fold_baseline_js_divs, ddof=1) if len(fold_baseline_js_divs) > 1 else None
        else:
            overall_baseline_js_div_mean = overall_baseline_js_div_std = None

        if len(fold_intervention_js_divs) > 0:
            overall_intervention_js_div_mean = np.mean(fold_intervention_js_divs)
            overall_intervention_js_div_std = np.std(fold_intervention_js_divs, ddof=1) if len(fold_intervention_js_divs) > 1 else None
        else:
            overall_intervention_js_div_mean = overall_intervention_js_div_std = None

        if len(fold_js_div_improvements) > 0:
            overall_js_div_improvement_mean = np.mean(fold_js_div_improvements)
            overall_js_div_improvement_std = np.std(fold_js_div_improvements, ddof=1) if len(fold_js_div_improvements) > 1 else None
        else:
            overall_js_div_improvement_mean = overall_js_div_improvement_std = None

        # Total Variation
        if len(fold_baseline_total_vars) > 0:
            overall_baseline_total_var_mean = np.mean(fold_baseline_total_vars)
            overall_baseline_total_var_std = np.std(fold_baseline_total_vars, ddof=1) if len(fold_baseline_total_vars) > 1 else None
        else:
            overall_baseline_total_var_mean = overall_baseline_total_var_std = None

        if len(fold_intervention_total_vars) > 0:
            overall_intervention_total_var_mean = np.mean(fold_intervention_total_vars)
            overall_intervention_total_var_std = np.std(fold_intervention_total_vars, ddof=1) if len(fold_intervention_total_vars) > 1 else None
        else:
            overall_intervention_total_var_mean = overall_intervention_total_var_std = None

        if len(fold_total_var_improvements) > 0:
            overall_total_var_improvement_mean = np.mean(fold_total_var_improvements)
            overall_total_var_improvement_std = np.std(fold_total_var_improvements, ddof=1) if len(fold_total_var_improvements) > 1 else None
        else:
            overall_total_var_improvement_mean = overall_total_var_improvement_std = None

        # Macro F1
        if len(fold_baseline_macro_f1s) > 0:
            overall_baseline_macro_f1_mean = np.mean(fold_baseline_macro_f1s)
            overall_baseline_macro_f1_std = np.std(fold_baseline_macro_f1s, ddof=1) if len(fold_baseline_macro_f1s) > 1 else None
        else:
            overall_baseline_macro_f1_mean = overall_baseline_macro_f1_std = None

        if len(fold_intervention_macro_f1s) > 0:
            overall_intervention_macro_f1_mean = np.mean(fold_intervention_macro_f1s)
            overall_intervention_macro_f1_std = np.std(fold_intervention_macro_f1s, ddof=1) if len(fold_intervention_macro_f1s) > 1 else None
        else:
            overall_intervention_macro_f1_mean = overall_intervention_macro_f1_std = None

        if len(fold_macro_f1_improvements) > 0:
            overall_macro_f1_improvement_mean = np.mean(fold_macro_f1_improvements)
            overall_macro_f1_improvement_std = np.std(fold_macro_f1_improvements, ddof=1) if len(fold_macro_f1_improvements) > 1 else None
        else:
            overall_macro_f1_improvement_mean = overall_macro_f1_improvement_std = None

        # Balanced Accuracy
        if len(fold_baseline_balanced_accs) > 0:
            overall_baseline_balanced_acc_mean = np.mean(fold_baseline_balanced_accs)
            overall_baseline_balanced_acc_std = np.std(fold_baseline_balanced_accs, ddof=1) if len(fold_baseline_balanced_accs) > 1 else None
        else:
            overall_baseline_balanced_acc_mean = overall_baseline_balanced_acc_std = None

        if len(fold_intervention_balanced_accs) > 0:
            overall_intervention_balanced_acc_mean = np.mean(fold_intervention_balanced_accs)
            overall_intervention_balanced_acc_std = np.std(fold_intervention_balanced_accs, ddof=1) if len(fold_intervention_balanced_accs) > 1 else None
        else:
            overall_intervention_balanced_acc_mean = overall_intervention_balanced_acc_std = None

        if len(fold_balanced_acc_improvements) > 0:
            overall_balanced_acc_improvement_mean = np.mean(fold_balanced_acc_improvements)
            overall_balanced_acc_improvement_std = np.std(fold_balanced_acc_improvements, ddof=1) if len(fold_balanced_acc_improvements) > 1 else None
        else:
            overall_balanced_acc_improvement_mean = overall_balanced_acc_improvement_std = None

        # Cohen's Kappa
        if len(fold_baseline_cohens) > 0:
            overall_baseline_cohen_mean = np.mean(fold_baseline_cohens)
            overall_baseline_cohen_std = np.std(fold_baseline_cohens, ddof=1) if len(fold_baseline_cohens) > 1 else None
        else:
            overall_baseline_cohen_mean = overall_baseline_cohen_std = None

        if len(fold_intervention_cohens) > 0:
            overall_intervention_cohen_mean = np.mean(fold_intervention_cohens)
            overall_intervention_cohen_std = np.std(fold_intervention_cohens, ddof=1) if len(fold_intervention_cohens) > 1 else None
        else:
            overall_intervention_cohen_mean = overall_intervention_cohen_std = None

        if len(fold_cohen_improvements) > 0:
            overall_cohen_improvement_mean = np.mean(fold_cohen_improvements)
            overall_cohen_improvement_std = np.std(fold_cohen_improvements, ddof=1) if len(fold_cohen_improvements) > 1 else None
        else:
            overall_cohen_improvement_mean = overall_cohen_improvement_std = None

        # Distribution Quality Score
        if len(fold_baseline_dist_qualities) > 0:
            overall_baseline_dist_quality_mean = np.mean(fold_baseline_dist_qualities)
            overall_baseline_dist_quality_std = np.std(fold_baseline_dist_qualities, ddof=1) if len(fold_baseline_dist_qualities) > 1 else None
        else:
            overall_baseline_dist_quality_mean = overall_baseline_dist_quality_std = None

        if len(fold_intervention_dist_qualities) > 0:
            overall_intervention_dist_quality_mean = np.mean(fold_intervention_dist_qualities)
            overall_intervention_dist_quality_std = np.std(fold_intervention_dist_qualities, ddof=1) if len(fold_intervention_dist_qualities) > 1 else None
        else:
            overall_intervention_dist_quality_mean = overall_intervention_dist_quality_std = None

        if len(fold_dist_quality_improvements) > 0:
            overall_dist_quality_improvement_mean = np.mean(fold_dist_quality_improvements)
            overall_dist_quality_improvement_std = np.std(fold_dist_quality_improvements, ddof=1) if len(fold_dist_quality_improvements) > 1 else None
        else:
            overall_dist_quality_improvement_mean = overall_dist_quality_improvement_std = None

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
                # Advanced metrics - Entropy
                'baseline_entropy_mean': overall_baseline_entropy_mean,
                'baseline_entropy_std': overall_baseline_entropy_std,
                'intervention_entropy_mean': overall_intervention_entropy_mean,
                'intervention_entropy_std': overall_intervention_entropy_std,
                'entropy_improvement_mean': overall_entropy_improvement_mean,
                'entropy_improvement_std': overall_entropy_improvement_std,
                # Advanced metrics - Gini Diversity
                'baseline_gini_diversity_mean': overall_baseline_gini_mean,
                'baseline_gini_diversity_std': overall_baseline_gini_std,
                'intervention_gini_diversity_mean': overall_intervention_gini_mean,
                'intervention_gini_diversity_std': overall_intervention_gini_std,
                'gini_diversity_improvement_mean': overall_gini_improvement_mean,
                'gini_diversity_improvement_std': overall_gini_improvement_std,
                # Advanced metrics - JS Divergence
                'baseline_js_divergence_mean': overall_baseline_js_div_mean,
                'baseline_js_divergence_std': overall_baseline_js_div_std,
                'intervention_js_divergence_mean': overall_intervention_js_div_mean,
                'intervention_js_divergence_std': overall_intervention_js_div_std,
                'js_divergence_improvement_mean': overall_js_div_improvement_mean,
                'js_divergence_improvement_std': overall_js_div_improvement_std,
                # Advanced metrics - Total Variation
                'baseline_total_variation_mean': overall_baseline_total_var_mean,
                'baseline_total_variation_std': overall_baseline_total_var_std,
                'intervention_total_variation_mean': overall_intervention_total_var_mean,
                'intervention_total_variation_std': overall_intervention_total_var_std,
                'total_variation_improvement_mean': overall_total_var_improvement_mean,
                'total_variation_improvement_std': overall_total_var_improvement_std,
                # Advanced metrics - Macro F1
                'baseline_macro_f1_mean': overall_baseline_macro_f1_mean,
                'baseline_macro_f1_std': overall_baseline_macro_f1_std,
                'intervention_macro_f1_mean': overall_intervention_macro_f1_mean,
                'intervention_macro_f1_std': overall_intervention_macro_f1_std,
                'macro_f1_improvement_mean': overall_macro_f1_improvement_mean,
                'macro_f1_improvement_std': overall_macro_f1_improvement_std,
                # Advanced metrics - Balanced Accuracy
                'baseline_balanced_accuracy_mean': overall_baseline_balanced_acc_mean,
                'baseline_balanced_accuracy_std': overall_baseline_balanced_acc_std,
                'intervention_balanced_accuracy_mean': overall_intervention_balanced_acc_mean,
                'intervention_balanced_accuracy_std': overall_intervention_balanced_acc_std,
                'balanced_accuracy_improvement_mean': overall_balanced_acc_improvement_mean,
                'balanced_accuracy_improvement_std': overall_balanced_acc_improvement_std,
                # Advanced metrics - Cohen's Kappa
                'baseline_cohen_kappa_mean': overall_baseline_cohen_mean,
                'baseline_cohen_kappa_std': overall_baseline_cohen_std,
                'intervention_cohen_kappa_mean': overall_intervention_cohen_mean,
                'intervention_cohen_kappa_std': overall_intervention_cohen_std,
                'cohen_kappa_improvement_mean': overall_cohen_improvement_mean,
                'cohen_kappa_improvement_std': overall_cohen_improvement_std,
                # Advanced metrics - Distribution Quality Score
                'baseline_dist_quality_score_mean': overall_baseline_dist_quality_mean,
                'baseline_dist_quality_score_std': overall_baseline_dist_quality_std,
                'intervention_dist_quality_score_mean': overall_intervention_dist_quality_mean,
                'intervention_dist_quality_score_std': overall_intervention_dist_quality_std,
                'dist_quality_score_improvement_mean': overall_dist_quality_improvement_mean,
                'dist_quality_score_improvement_std': overall_dist_quality_improvement_std,
                'per_fold_aggregates': {
                    'baseline': fold_baseline_accs,
                    'intervention': fold_intervention_accs,
                    'improvement': fold_improvements,
                    'baseline_kendall': fold_baseline_kendalls,
                    'intervention_kendall': fold_intervention_kendalls,
                    'kendall_improvement': fold_kendall_improvements,
                    # Advanced metrics per fold
                    'baseline_entropy': fold_baseline_entropies,
                    'intervention_entropy': fold_intervention_entropies,
                    'entropy_improvement': fold_entropy_improvements,
                    'baseline_gini_diversity': fold_baseline_ginis,
                    'intervention_gini_diversity': fold_intervention_ginis,
                    'gini_diversity_improvement': fold_gini_improvements,
                    'baseline_js_divergence': fold_baseline_js_divs,
                    'intervention_js_divergence': fold_intervention_js_divs,
                    'js_divergence_improvement': fold_js_div_improvements,
                    'baseline_total_variation': fold_baseline_total_vars,
                    'intervention_total_variation': fold_intervention_total_vars,
                    'total_variation_improvement': fold_total_var_improvements,
                    'baseline_macro_f1': fold_baseline_macro_f1s,
                    'intervention_macro_f1': fold_intervention_macro_f1s,
                    'macro_f1_improvement': fold_macro_f1_improvements,
                    'baseline_balanced_accuracy': fold_baseline_balanced_accs,
                    'intervention_balanced_accuracy': fold_intervention_balanced_accs,
                    'balanced_accuracy_improvement': fold_balanced_acc_improvements,
                    'baseline_cohen_kappa': fold_baseline_cohens,
                    'intervention_cohen_kappa': fold_intervention_cohens,
                    'cohen_kappa_improvement': fold_cohen_improvements,
                    'baseline_dist_quality': fold_baseline_dist_qualities,
                    'intervention_dist_quality': fold_intervention_dist_qualities,
                    'dist_quality_improvement': fold_dist_quality_improvements
                },
                'n_folds': n_folds,
                'n_baseline_kendall_folds': len(fold_baseline_kendalls),
                'n_intervention_kendall_folds': len(fold_intervention_kendalls),
                'n_kendall_improvement_folds': len(fold_kendall_improvements),
                # Counts for advanced metrics
                'n_baseline_entropy_folds': len(fold_baseline_entropies),
                'n_intervention_entropy_folds': len(fold_intervention_entropies),
                'n_baseline_gini_folds': len(fold_baseline_ginis),
                'n_intervention_gini_folds': len(fold_intervention_ginis),
                'n_baseline_js_div_folds': len(fold_baseline_js_divs),
                'n_intervention_js_div_folds': len(fold_intervention_js_divs),
                'n_baseline_total_var_folds': len(fold_baseline_total_vars),
                'n_intervention_total_var_folds': len(fold_intervention_total_vars),
                'n_baseline_macro_f1_folds': len(fold_baseline_macro_f1s),
                'n_intervention_macro_f1_folds': len(fold_intervention_macro_f1s),
                'n_baseline_balanced_acc_folds': len(fold_baseline_balanced_accs),
                'n_intervention_balanced_acc_folds': len(fold_intervention_balanced_accs),
                'n_baseline_cohen_folds': len(fold_baseline_cohens),
                'n_intervention_cohen_folds': len(fold_intervention_cohens),
                'n_baseline_dist_quality_folds': len(fold_baseline_dist_qualities),
                'n_intervention_dist_quality_folds': len(fold_intervention_dist_qualities)
            },
            'timestamp': datetime.now().isoformat()
        }

        result_file = results_dir / f"{demographic}_{args.probe_type}_k{args.top_k_heads}_s{args.intervention_strength}_{args.run_id}_intervention_results.pkl"
        with open(result_file, 'wb') as f:
            pickle.dump(demographic_results, f)

        print(f"\nSaved results: {result_file}")

        all_demographic_results[demographic] = demographic_results

    # Save aggregate results
    print(f"\n{'='*80}")
    print("INTERVENTION PHASE COMPLETE")
    print(f"{'='*80}")

    summary_file = results_dir / f'intervention_summary_{args.probe_type}_k{args.top_k_heads}_s{args.intervention_strength}_{args.run_id}.json'
    summary_data = {
        'config': vars(args),
        'demographics': {
            demo: {
                # Basic accuracy metrics
                'baseline_accuracy_mean': results['overall_metrics']['baseline_accuracy_mean'],
                'baseline_accuracy_std': results['overall_metrics']['baseline_accuracy_std'],
                'intervention_accuracy_mean': results['overall_metrics']['intervention_accuracy_mean'],
                'intervention_accuracy_std': results['overall_metrics']['intervention_accuracy_std'],
                'improvement_mean': results['overall_metrics']['improvement_mean'],
                'improvement_std': results['overall_metrics']['improvement_std'],

                # Kendall's tau (ordinal correlation)
                'baseline_kendall_tau_mean': results['overall_metrics']['baseline_kendall_tau_mean'],
                'baseline_kendall_tau_std': results['overall_metrics']['baseline_kendall_tau_std'],
                'intervention_kendall_tau_mean': results['overall_metrics']['intervention_kendall_tau_mean'],
                'intervention_kendall_tau_std': results['overall_metrics']['intervention_kendall_tau_std'],
                'kendall_improvement_mean': results['overall_metrics']['kendall_improvement_mean'],
                'kendall_improvement_std': results['overall_metrics']['kendall_improvement_std'],

                # Advanced metrics - Diversity
                'baseline_entropy_mean': results['overall_metrics'].get('baseline_entropy_mean'),
                'baseline_entropy_std': results['overall_metrics'].get('baseline_entropy_std'),
                'intervention_entropy_mean': results['overall_metrics'].get('intervention_entropy_mean'),
                'intervention_entropy_std': results['overall_metrics'].get('intervention_entropy_std'),
                'entropy_improvement_mean': results['overall_metrics'].get('entropy_improvement_mean'),
                'entropy_improvement_std': results['overall_metrics'].get('entropy_improvement_std'),

                'baseline_gini_diversity_mean': results['overall_metrics'].get('baseline_gini_diversity_mean'),
                'baseline_gini_diversity_std': results['overall_metrics'].get('baseline_gini_diversity_std'),
                'intervention_gini_diversity_mean': results['overall_metrics'].get('intervention_gini_diversity_mean'),
                'intervention_gini_diversity_std': results['overall_metrics'].get('intervention_gini_diversity_std'),
                'gini_diversity_improvement_mean': results['overall_metrics'].get('gini_diversity_improvement_mean'),
                'gini_diversity_improvement_std': results['overall_metrics'].get('gini_diversity_improvement_std'),

                # Advanced metrics - Distributional similarity
                'baseline_js_divergence_mean': results['overall_metrics'].get('baseline_js_divergence_mean'),
                'baseline_js_divergence_std': results['overall_metrics'].get('baseline_js_divergence_std'),
                'intervention_js_divergence_mean': results['overall_metrics'].get('intervention_js_divergence_mean'),
                'intervention_js_divergence_std': results['overall_metrics'].get('intervention_js_divergence_std'),
                'js_divergence_improvement_mean': results['overall_metrics'].get('js_divergence_improvement_mean'),
                'js_divergence_improvement_std': results['overall_metrics'].get('js_divergence_improvement_std'),

                'baseline_total_variation_mean': results['overall_metrics'].get('baseline_total_variation_mean'),
                'baseline_total_variation_std': results['overall_metrics'].get('baseline_total_variation_std'),
                'intervention_total_variation_mean': results['overall_metrics'].get('intervention_total_variation_mean'),
                'intervention_total_variation_std': results['overall_metrics'].get('intervention_total_variation_std'),
                'total_variation_improvement_mean': results['overall_metrics'].get('total_variation_improvement_mean'),
                'total_variation_improvement_std': results['overall_metrics'].get('total_variation_improvement_std'),

                # Advanced metrics - Multiclass performance
                'baseline_macro_f1_mean': results['overall_metrics'].get('baseline_macro_f1_mean'),
                'baseline_macro_f1_std': results['overall_metrics'].get('baseline_macro_f1_std'),
                'intervention_macro_f1_mean': results['overall_metrics'].get('intervention_macro_f1_mean'),
                'intervention_macro_f1_std': results['overall_metrics'].get('intervention_macro_f1_std'),
                'macro_f1_improvement_mean': results['overall_metrics'].get('macro_f1_improvement_mean'),
                'macro_f1_improvement_std': results['overall_metrics'].get('macro_f1_improvement_std'),

                'baseline_balanced_accuracy_mean': results['overall_metrics'].get('baseline_balanced_accuracy_mean'),
                'baseline_balanced_accuracy_std': results['overall_metrics'].get('baseline_balanced_accuracy_std'),
                'intervention_balanced_accuracy_mean': results['overall_metrics'].get('intervention_balanced_accuracy_mean'),
                'intervention_balanced_accuracy_std': results['overall_metrics'].get('intervention_balanced_accuracy_std'),
                'balanced_accuracy_improvement_mean': results['overall_metrics'].get('balanced_accuracy_improvement_mean'),
                'balanced_accuracy_improvement_std': results['overall_metrics'].get('balanced_accuracy_improvement_std'),

                'baseline_cohen_kappa_mean': results['overall_metrics'].get('baseline_cohen_kappa_mean'),
                'baseline_cohen_kappa_std': results['overall_metrics'].get('baseline_cohen_kappa_std'),
                'intervention_cohen_kappa_mean': results['overall_metrics'].get('intervention_cohen_kappa_mean'),
                'intervention_cohen_kappa_std': results['overall_metrics'].get('intervention_cohen_kappa_std'),
                'cohen_kappa_improvement_mean': results['overall_metrics'].get('cohen_kappa_improvement_mean'),
                'cohen_kappa_improvement_std': results['overall_metrics'].get('cohen_kappa_improvement_std'),

                # Combined quality score
                'baseline_dist_quality_score_mean': results['overall_metrics'].get('baseline_dist_quality_score_mean'),
                'baseline_dist_quality_score_std': results['overall_metrics'].get('baseline_dist_quality_score_std'),
                'intervention_dist_quality_score_mean': results['overall_metrics'].get('intervention_dist_quality_score_mean'),
                'intervention_dist_quality_score_std': results['overall_metrics'].get('intervention_dist_quality_score_std'),
                'dist_quality_score_improvement_mean': results['overall_metrics'].get('dist_quality_score_improvement_mean'),
                'dist_quality_score_improvement_std': results['overall_metrics'].get('dist_quality_score_improvement_std'),

                # Metadata
                'n_folds': results['n_folds'],
                'n_baseline_kendall_folds': results['overall_metrics']['n_baseline_kendall_folds'],
                'n_intervention_kendall_folds': results['overall_metrics']['n_intervention_kendall_folds'],
                'n_kendall_improvement_folds': results['overall_metrics']['n_kendall_improvement_folds']
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
        print(f"  {demo}: {metrics['baseline_accuracy_mean']*100:.1f}% -> "
              f"{metrics['intervention_accuracy_mean']*100:.1f}% "
              f"({metrics['improvement_mean']:+.1f} points)")


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
