# Demographic Circuit Interventions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for extracting and testing demographic circuits in language models using ANES surveys. This toolkit enables robust measurement of how well LLMs encode demographic information and how interventions on those circuits affect model predictions.

## Features

- **Two-Phase Design**: Separate extraction and intervention phases for efficiency
- **Memory-Optimized Extraction**: Probes per-fold and saves only top N components (90%+ file size reduction)
- **No Data Leakage**: K-fold probing ensures component selection never sees test questions
- **K-Fold Cross-Validation**: Robust evaluation across multiple train/test splits
- **Multiple Probe Types**: Support for attention heads and MLP layers
- **Comprehensive Analysis**: Saves probing results (CSV/JSON) and Spearman correlation visualizations
- **Flexible Configuration**: Fully configurable via command-line arguments
- **Run Management**: Unique run identifiers prevent overwriting results
- **Detailed Metrics**: Statistical measures with per-fold analysis
- **Based on ANES Data**: Uses real survey data from the American National Election Studies

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended) or CPU
- 16GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RobustDemographicCircuits.git
cd RobustDemographicCircuits
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download ANES data (required):
   - Download ANES 2024 data from [ANES website](https://electionstudies.org/)
   - Place CSV in `data/` directory or specify path with `--anes_data_path`

## Quick Start

### Run Complete Experiment

Test demographic circuits with default settings:

```bash
python robust_demographic_experiment.py \
    --model meta-llama/Llama-3.2-1B \
    --demographics gender age \
    --n_folds 4 \
    --top_k_heads 10
```

This will:
1. Extract activations for all political questions
2. Perform k-fold probing (select top 10 components per fold from train questions only)
3. Run 4-fold cross-validation on interventions
4. Generate comprehensive results, probing results (CSV/JSON), and visualizations (PNG)

### Two-Phase Workflow (Recommended)

**Phase 1: Extract activations with k-fold probing** (slow, run once)

```bash
python robust_demographic_experiment.py \
    --phase extract \
    --model meta-llama/Llama-3.2-1B \
    --probe_type attention \
    --demographics gender age race education \
    --n_folds 5 \
    --top_k_heads 10 \
    --run_id my_experiment_1
```

This creates:
- 5 fold-specific extraction files per demographic (memory-optimized)
- Probing results CSV/JSON files per fold
- Spearman correlation visualizations per fold

**Phase 2: Test interventions** (fast, iterate multiple times)

```bash
python robust_demographic_experiment.py \
    --phase intervene \
    --model meta-llama/Llama-3.2-1B \
    --probe_type attention \
    --demographics gender age race education \
    --intervention_strength 10.0 \
    --n_folds 5 \
    --run_id my_experiment_1
```

**Note**: Must use same `--n_folds` and `--run_id` as extraction phase!

## Usage

### Basic Commands

```bash
# Test all demographics with default settings
python robust_demographic_experiment.py

# Test specific demographics
python robust_demographic_experiment.py --demographics gender age race

# Use MLP layers instead of attention heads
python robust_demographic_experiment.py --probe_type mlp --intervention_strength 5.0

# Increase number of folds for more robust estimates
python robust_demographic_experiment.py --n_folds 10

# Use different model
python robust_demographic_experiment.py --model meta-llama/Llama-3.2-3B
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | HuggingFace model name | `meta-llama/Llama-3.2-1B` |
| `--probe_type` | Probe type: `attention`, `mlp`, or `both` | `attention` |
| `--phase` | Phase: `extract`, `intervene`, or `both` | `both` |
| `--demographics` | Demographics to test (space-separated) | All 9 demographics |
| `--n_folds` | Number of k-fold splits (MUST match between phases) | `4` |
| `--top_k_heads` | Number of top components to save per fold | `10` |
| `--intervention_strength` | Intervention alpha parameter | `10.0` |
| `--n_samples_per_category` | Samples per category for extraction | `100` |
| `--run_id` | Unique identifier for this run (auto-generated if not specified) | Timestamp |
| `--ridge_alpha` | Ridge regression regularization parameter | `1.0` |
| `--eval_sample_size` | Sample size per test question during intervention | `50` |

See full list with `python robust_demographic_experiment.py --help`

## Output

### Directory Structure

```
robust_experiment_results/
├── extractions/
│   ├── gender_attention_20250113_143022_fold1_extractions.pkl
│   ├── gender_attention_20250113_143022_fold2_extractions.pkl
│   ├── gender_attention_20250113_143022_fold3_extractions.pkl
│   ├── gender_attention_20250113_143022_fold4_extractions.pkl
│   ├── age_attention_20250113_143022_fold1_extractions.pkl
│   ├── ...
│   └── probing_results/
│       ├── gender_fold1_attention_probing_results.csv
│       ├── gender_fold1_attention_probing_results.json
│       ├── gender_fold1_attention_spearman_correlations.png
│       ├── gender_fold2_attention_probing_results.csv
│       ├── gender_fold2_attention_spearman_correlations.png
│       └── ...
└── intervention_results/
    ├── gender_attention_20250113_143022_intervention_results.pkl
    ├── age_attention_20250113_143022_intervention_results.pkl
    └── intervention_summary.json
```

### Probing Results Files

**CSV Format** (`*_probing_results.csv`):
```csv
layer,head,accuracy,spearman_r,p_value
15,28,0.72,0.65,0.0001
14,12,0.68,0.58,0.0003
...
```

**Visualizations** (`*_spearman_correlations.png`):
- **For Attention**:
  - Bar plot showing top 50 heads by Spearman correlation (top K highlighted in red)
  - Heatmap showing correlations across all layers and heads (top K marked with stars)
- **For MLP**:
  - Bar plot of Spearman correlations by layer
  - Bar plot of classification accuracies by layer

### Intervention Results Format

The `intervention_summary.json` contains a **list** of all runs (appends new results):

```json
[
  {
    "config": {
      "model": "meta-llama/Llama-3.2-1B",
      "probe_type": "attention",
      "n_folds": 4,
      "top_k_heads": 10,
      "intervention_strength": 10.0,
      "run_id": "20250113_143022"
    },
    "demographics": {
      "gender": {
        "baseline_accuracy": 0.65,
        "intervention_accuracy": 0.72,
        "improvement": 7.0,
        "n_folds": 4
      },
      ...
    },
    "timestamp": "2025-01-13T14:30:22"
  },
  ...
]
```

## Analyzing Probing Results

After running extraction, you can analyze which components encode demographic information:

### CSV Analysis

```python
import pandas as pd

# Load probing results
df = pd.read_csv('robust_experiment_results/extractions/probing_results/gender_fold1_attention_probing_results.csv')

# Top 10 heads
print(df.head(10))

# Filter by significance
significant = df[df['p_value'] < 0.001]
print(f"Found {len(significant)} significant heads")

# Analyze by layer
layer_stats = df.groupby('layer')['spearman_r'].agg(['mean', 'max', 'count'])
print(layer_stats)
```

### Visualization Analysis

Open the generated PNG files to:
- Identify which layers/heads encode demographic information
- See the strength of encoding (via Spearman correlation)
- Compare encoding patterns across folds (stability analysis)
- Understand if encoding is concentrated in early, middle, or late layers

### Cross-Fold Comparison

```python
import pandas as pd

# Load all folds
folds = []
for i in range(1, 5):
    df = pd.read_csv(f'robust_experiment_results/extractions/probing_results/gender_fold{i}_attention_probing_results.csv')
    df['fold'] = i
    folds.append(df)

all_folds = pd.concat(folds)

# See which heads appear in top 10 across all folds
top10_per_fold = all_folds.groupby('fold').head(10)
head_counts = top10_per_fold.groupby(['layer', 'head']).size()
stable_heads = head_counts[head_counts >= 3]  # Appears in at least 3 folds
print("Stable heads across folds:")
print(stable_heads)
```

## Examples

Common usage patterns:

### Quick Test (Single Demographic)
```bash
python robust_demographic_experiment.py \
    --demographics gender \
    --n_folds 3 \
    --top_k_heads 5
```

### Full Experiment (All Demographics)
```bash
python robust_demographic_experiment.py \
    --demographics gender age race education marital_status income ideology religion urban_rural \
    --n_folds 5 \
    --top_k_heads 10 \
    --run_id full_experiment_v1
```

### Compare Intervention Strengths
```bash
# Extract once
python robust_demographic_experiment.py \
    --phase extract \
    --demographics gender \
    --n_folds 4 \
    --run_id strength_comparison

# Try different strengths
for strength in 5.0 10.0 50.0 100.0; do
    python robust_demographic_experiment.py \
        --phase intervene \
        --demographics gender \
        --intervention_strength $strength \
        --n_folds 4 \
        --run_id strength_comparison
done
```

### MLP vs Attention Comparison
```bash
# Test attention heads
python robust_demographic_experiment.py \
    --probe_type attention \
    --demographics gender age \
    --run_id attention_test

# Test MLP layers
python robust_demographic_experiment.py \
    --probe_type mlp \
    --demographics gender age \
    --run_id mlp_test
```

## Methodology

### K-Fold Cross-Validation with No Data Leakage

Questions are split into k folds. For each fold:

1. **Extract**: Extract full activations for ALL questions (temporary)
2. **Probe**: Concatenate activations from **train questions only** (k-1 folds)
3. **Select**: Train ridge classifiers and identify top-k components with strongest predictive power
4. **Filter**: Keep only top-k components for all questions, discard the rest
5. **Save**: Save fold-specific extraction file with filtered activations
6. **Intervene**: During intervention phase, load fold-specific extraction and test on held-out fold
7. **Aggregate**: Average metrics across all folds with standard deviations

**Critical**: Component selection uses ONLY training questions, ensuring no data leakage into test folds.

### Memory Optimization

The framework dramatically reduces memory usage:

- **Without optimization**: Saves all 256 attention heads (16 layers × 16 heads) or 16 MLP layers
- **With optimization**: Saves only top K components (default K=10)
- **Memory savings**: ~96% reduction for attention, ~37.5% for MLP
- **Per-fold selection**: Different folds may select different components based on their training data

### Circuit Extraction

The framework uses linear probing to identify attention heads or MLP layers that encode demographic information:

1. Generate prompts with demographic attributes **excluded** (to measure implicit encoding)
2. Extract activations from the model for all questions
3. For each fold:
   - Concatenate activations from train questions only
   - Train ridge classifiers to predict demographics from activations
   - Rank heads/layers by Spearman correlation
   - Select top-k components
   - Save probing results (CSV/JSON) and visualizations (PNG)

### Intervention

Selected circuits are then intervened upon during inference:

```
activation_new = activation_old + α * direction * std
```

Where:
- `α` is the intervention strength (configurable via `--intervention_strength`)
- `direction` is the learned probe coefficient (from ridge classifier)
- `std` normalizes by feature scale (prevents numerical instability)

## Supported Demographics

- `gender` - Binary: Man/Woman
- `age` - Three categories: Young Adult/Adult/Senior
- `race` - Multiple categories
- `education` - Three levels: Low/Medium/High
- `marital_status` - Three categories
- `income` - Three levels: Low/Middle/High
- `ideology` - Three categories: Left/Center/Right
- `religion` - Two categories: Religious/Not Religious
- `urban_rural` - Two categories: Urban/Rural

## Political Questions

The framework tests interventions on 12 political opinion questions:

- Abortion
- Death penalty
- Military force
- Defense spending
- Government jobs
- Government help to Black Americans
- College opinions
- DEI opinions
- Journalist access
- Transgender bathrooms
- Birthright citizenship
- Immigration policy


## Tips and Best Practices

### Memory Management

- **Start small**: Test with 1-2 demographics first before scaling up
- **Adjust top_k**: Reduce `--top_k_heads` if memory is tight (minimum 5 recommended)
- **Use CPU for small models**: For models < 3B parameters, CPU may be sufficient

### Choosing Parameters

- **n_folds**:
  - Use 3-5 for quick experiments
  - Use 5-10 for publication-quality results
  - More folds = more robust but slower

- **top_k_heads**:
  - 10 is a good default balance
  - Increase to 15-20 if you have many questions
  - Decrease to 5 for very small datasets

- **intervention_strength**:
  - Start with 10.0 for attention heads
  - Start with 5.0 for MLP layers
  - Too high = model collapse, too low = no effect
  - Use the strength comparison example to find optimal value

### Reproducibility

- **Always specify** `--run_id` for important experiments
- **Keep** `--n_folds` and `--seed` consistent across runs
- **Document** all parameters in your experiment log
- **Save** the `intervention_summary.json` for all runs

### Analysis Workflow

1. Run extraction phase once per configuration
2. Analyze probing results (CSV/PNG files) to understand which components matter
3. Run multiple intervention experiments with different strengths
4. Compare results across demographics using the summary JSON
5. Visualize stability by comparing top components across folds

## Related Work

This framework builds on:

- **ANES Data**: American National Election Studies surveys
- **Circuit Analysis**: Mechanistic interpretability for language models
- **Activation Steering**: Intervention-based causal analysis
- **Baukit**: Activation extraction toolkit
- **Ridge Regression Probing**: Linear probing for feature attribution
- **K-Fold Cross-Validation**: Statistical evaluation methodology

## License

MIT License - see LICENSE file for details



