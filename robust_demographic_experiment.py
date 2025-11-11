"""
Robust Demographic Intervention Experiment with K-Fold Validation

This script runs comprehensive demographic intervention experiments with:
1. Separate extraction and intervention phases
2. K-fold cross-validation on political questions
3. Flexible configuration via command-line arguments
4. Comprehensive result tracking and reporting

Usage:
    # Extract circuits first (saves activations for all questions)
    python robust_demographic_experiment.py --model meta-llama/Llama-3.2-1B \\
        --probe_type attention --phase extract --demographics gender age race

    # Run k-fold validation on interventions
    python robust_demographic_experiment.py --model meta-llama/Llama-3.2-1B \\
        --probe_type attention --phase intervene --demographics gender age race \\
        --intervention_strength 100.0 --top_k_heads 10 --n_folds 5

    # Run both phases sequentially
    python robust_demographic_experiment.py --model meta-llama/Llama-3.2-1B \\
        --probe_type attention --phase both --demographics gender age
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
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
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


def run_extraction_phase(args):
    """
    Phase 1: Extract activations for all questions and save to disk.

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

    # Process each demographic
    for demographic in args.demographics:
        print(f"\n{'='*80}")
        print(f"EXTRACTING: {demographic.upper()}")
        print(f"{'='*80}")

        # Check if already extracted
        extraction_file = output_dir / f"{demographic}_{args.probe_type}_extractions.pkl"
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

        # Save extractions
        extraction_data = {
            'demographic': demographic,
            'probe_type': args.probe_type,
            'model': args.model,
            'n_samples_per_category': args.n_samples_per_category,
            'question_extractions': question_extractions,
            'timestamp': datetime.now().isoformat()
        }

        with open(extraction_file, 'wb') as f:
            pickle.dump(extraction_data, f)

        print(f"\nSaved extractions for {len(question_extractions)} questions: {extraction_file}")

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
        std = np.std(coef) if coef is not None else 0.0
        intercept = 0.0
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

    # Setup intervention engine
    if probe_type == 'attention':
        engine = CircuitInterventionEngine(model, intervention_weights, device)
        ConfigClass = InterventionConfig
        config_param = 'top_k_heads'
    else:  # mlp
        engine = MLPInterventionEngine(model, intervention_weights, device)
        ConfigClass = MLPInterventionConfig
        config_param = 'top_k_layers'

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

        baseline_predictions = []
        intervention_predictions = []
        true_labels = []

        for idx, user_profile in test_users.iterrows():
            prompt = create_prompt(user_profile, question, answer_options=answer_options, answer=None)

            # Baseline prediction
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                baseline_logits = outputs.logits[:, -1, :].cpu()

            baseline_pred = predict_from_logits(baseline_logits, answer_options, tokenizer)
            baseline_predictions.append(baseline_pred)

            # Intervention prediction
            user_category = user_profile[demographic_attr]
            intervention_direction = 'maximize' if user_category == category_names[0] else 'minimize'

            config_kwargs = {
                'intervention_strength': intervention_strength,
                config_param: len(intervention_weights),
                'intervention_direction': intervention_direction
            }
            config = ConfigClass(**config_kwargs)

            intervention_logits = engine.intervene_activation_steering_logits(
                prompt, tokenizer, config
            )

            intervention_pred = predict_from_logits(intervention_logits.cpu(), answer_options, tokenizer)
            intervention_predictions.append(intervention_pred)

            true_labels.append(user_profile[question])

        # Calculate accuracies
        baseline_acc = accuracy_score(true_labels, baseline_predictions)
        intervention_acc = accuracy_score(true_labels, intervention_predictions)
        improvement = (intervention_acc - baseline_acc) * 100

        test_results[question] = {
            'baseline_accuracy': baseline_acc,
            'intervention_accuracy': intervention_acc,
            'improvement': improvement,
            'n_samples': len(test_users)
        }

    return test_results


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

        # Load extraction
        extraction_file = extraction_dir / f"{demographic}_{args.probe_type}_extractions.pkl"
        if not extraction_file.exists():
            print(f"ERROR: Extraction file not found: {extraction_file}")
            print("Run extraction phase first!")
            continue

        print(f"Loading extractions from: {extraction_file}")
        with open(extraction_file, 'rb') as f:
            extraction_data = pickle.load(f)

        question_extractions = extraction_data['question_extractions']
        questions = list(question_extractions.keys())

        print(f"Loaded {len(questions)} questions")

        # Setup k-fold cross-validation
        kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

        fold_results = []

        for fold_idx, (train_indices, test_indices) in enumerate(kfold.split(questions)):
            print(f"\n{'-'*80}")
            print(f"FOLD {fold_idx + 1}/{args.n_folds}")
            print(f"{'-'*80}")

            train_questions = [questions[i] for i in train_indices]
            test_questions = [questions[i] for i in test_indices]

            print(f"Train questions ({len(train_questions)}): {train_questions}")
            print(f"Test questions ({len(test_questions)}): {test_questions}")

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

            # Train probes
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

                print(f"\nFold {fold_idx + 1} Results:")
                print(f"  Baseline:     {fold_baseline*100:.1f}%")
                print(f"  Intervention: {fold_intervention*100:.1f}%")
                print(f"  Improvement:  {fold_improvement:+.1f} points")

            fold_results.append({
                'fold': fold_idx,
                'train_questions': train_questions,
                'test_questions': test_questions,
                'test_results': test_results,
                'intervention_weights': intervention_weights
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

        # Average across folds for each question
        question_aggregates = {}
        for question, results_list in all_question_results.items():
            question_aggregates[question] = {
                'baseline_accuracy_mean': np.mean([r['baseline_accuracy'] for r in results_list]),
                'baseline_accuracy_std': np.std([r['baseline_accuracy'] for r in results_list]),
                'intervention_accuracy_mean': np.mean([r['intervention_accuracy'] for r in results_list]),
                'intervention_accuracy_std': np.std([r['intervention_accuracy'] for r in results_list]),
                'improvement_mean': np.mean([r['improvement'] for r in results_list]),
                'improvement_std': np.std([r['improvement'] for r in results_list]),
                'n_folds': len(results_list)
            }

        # Overall metrics
        overall_baseline = np.mean([v['baseline_accuracy_mean'] for v in question_aggregates.values()])
        overall_intervention = np.mean([v['intervention_accuracy_mean'] for v in question_aggregates.values()])
        overall_improvement = np.mean([v['improvement_mean'] for v in question_aggregates.values()])

        print(f"\nOverall Results (averaged across {args.n_folds} folds):")
        print(f"  Baseline:     {overall_baseline*100:.1f}%")
        print(f"  Intervention: {overall_intervention*100:.1f}%")
        print(f"  Improvement:  {overall_improvement:+.1f} points")

        # Save demographic results
        demographic_results = {
            'demographic': demographic,
            'category_names': category_names,
            'probe_type': args.probe_type,
            'n_folds': args.n_folds,
            'top_k_heads': args.top_k_heads,
            'intervention_strength': args.intervention_strength,
            'fold_results': fold_results,
            'question_aggregates': question_aggregates,
            'overall_metrics': {
                'baseline_accuracy': overall_baseline,
                'intervention_accuracy': overall_intervention,
                'improvement': overall_improvement
            },
            'timestamp': datetime.now().isoformat()
        }

        result_file = results_dir / f"{demographic}_{args.probe_type}_intervention_results.pkl"
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
                'baseline_accuracy': results['overall_metrics']['baseline_accuracy'],
                'intervention_accuracy': results['overall_metrics']['intervention_accuracy'],
                'improvement': results['overall_metrics']['improvement'],
                'n_folds': results['n_folds']
            }
            for demo, results in all_demographic_results.items()
        },
        'timestamp': datetime.now().isoformat()
    }

    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)

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
