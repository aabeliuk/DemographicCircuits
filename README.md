# Demographic Circuit Interventions

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for extracting and testing demographic circuits in language models using ANES surveys. This toolkit enables robust measurement of how well LLMs encode demographic information and how interventions on those circuits affect model predictions.

## Features

- **Two-Phase Design**: Separate extraction and intervention phases for efficiency
- **K-Fold Cross-Validation**: Robust evaluation across multiple train/test splits
- **Multiple Probe Types**: Support for attention heads and MLP layers
- **Flexible Configuration**: Fully configurable via command-line arguments
- **Comprehensive Reporting**: Detailed metrics with statistical measures
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
    --n_folds 4
```

This will:
1. Extract activations for all political questions
2. Run 4-fold cross-validation
3. Generate comprehensive results and visualizations

### Two-Phase Workflow (Recommended)

**Phase 1: Extract activations** (slow, run once)

```bash
python robust_demographic_experiment.py \
    --phase extract \
    --model meta-llama/Llama-3.2-1B \
    --probe_type attention \
    --demographics gender age race education
```

**Phase 2: Test interventions** (fast, iterate multiple times)

```bash
python robust_demographic_experiment.py \
    --phase intervene \
    --model meta-llama/Llama-3.2-1B \
    --probe_type attention \
    --demographics gender age race education \
    --intervention_strength 10.0 \
    --top_k_heads 10 \
    --n_folds 4
```

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
| `--demographics` | Demographics to test | All 9 demographics |
| `--n_folds` | Number of cross-validation folds | `4` |
| `--top_k_heads` | Number of top heads/layers to use | `10` |
| `--intervention_strength` | Intervention alpha parameter | `10.0` |
| `--n_samples_per_category` | Samples per category for extraction | `100` |

See full list with `python robust_demographic_experiment.py --help`

## Output

### Directory Structure

```
robust_experiment_results/
├── extractions/
│   ├── gender_attention_extractions.pkl
│   ├── age_attention_extractions.pkl
│   └── ...
└── intervention_results/
    ├── gender_attention_intervention_results.pkl
    ├── age_attention_intervention_results.pkl
    └── intervention_summary.json
```

### Results Format

The `intervention_summary.json` contains:

```json
{
  "config": {
    "model": "meta-llama/Llama-3.2-1B",
    "probe_type": "attention",
    "n_folds": 4,
    "top_k_heads": 10,
    "intervention_strength": 10.0
  },
  "demographics": {
    "gender": {
      "baseline_accuracy": 0.65,
      "intervention_accuracy": 0.72,
      "improvement": 7.0,
      "n_folds": 4
    },
    ...
  }
}
```

## Examples

See the `examples/` directory for common use cases:

- `examples/quick_test.sh` - Fast test with single demographic
- `examples/full_experiment.sh` - Comprehensive experiment with all demographics
- `examples/compare_models.sh` - Compare different LLM models
- `examples/sweep_parameters.sh` - Grid search over intervention parameters

## Methodology

### K-Fold Cross-Validation

Questions are split into k folds. For each fold:

1. **Train**: Extract activations from k-1 folds and train probes
2. **Test**: Evaluate interventions on the held-out fold
3. **Aggregate**: Average metrics across all folds with standard deviations

This prevents overfitting and provides robust performance estimates.

### Circuit Extraction

The framework uses linear probing to identify attention heads or MLP layers that encode demographic information:

1. Generate prompts with demographic attributes **excluded**
2. Extract activations from the model
3. Train ridge classifiers to predict demographics from activations
4. Select top-k heads/layers with strongest predictive power

### Intervention

Selected circuits are then intervened upon during inference:

```
activation_new = activation_old + α * direction * std
```

Where:
- `α` is the intervention strength
- `direction` is the learned probe coefficient
- `std` normalizes by feature scale

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


## Related Work

This framework builds on:

- **ANES Data**: American National Election Studies
- **Circuit Analysis**: Mechanistic interpretability for language models
- **Activation Steering**: Intervention-based causal analysis
- **Baukit**: Activation extraction toolkit

## License

MIT License - see LICENSE file for details



