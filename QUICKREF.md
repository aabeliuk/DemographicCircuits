# Quick Reference Guide

Essential commands and configurations for Robust Demographic Circuits.

## Installation

```bash
git clone <your-repo-url>
cd RobustDemographicCircuits
pip install -r requirements.txt
```

## Basic Commands

### Quick Test (5 minutes)
```bash
python robust_demographic_experiment.py \
    --demographics gender \
    --n_folds 2 \
    --n_samples_per_category 25
```

### Full Run (2-4 hours)
```bash
python robust_demographic_experiment.py
```

### Two-Phase Workflow
```bash
# Phase 1: Extract (slow, run once)
python robust_demographic_experiment.py --phase extract

# Phase 2: Intervene (fast, iterate)
python robust_demographic_experiment.py --phase intervene --intervention_strength 10.0
```

## Common Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--model` | Any HF model | `meta-llama/Llama-3.2-1B` | LLM to use |
| `--probe_type` | `attention`, `mlp`, `both` | `attention` | Type of circuit |
| `--phase` | `extract`, `intervene`, `both` | `both` | What to run |
| `--demographics` | `gender age race ...` | All 9 | Which demos |
| `--n_folds` | 2-10 | `4` | CV folds |
| `--top_k_heads` | 5-20 | `10` | # heads/layers |
| `--intervention_strength` | 5-20 | `10.0` | Alpha value |
| `--n_samples_per_category` | 25-200 | `100` | Extraction samples |

## Quick Modifications

### Test Different Models
```bash
python robust_demographic_experiment.py --model gpt2
python robust_demographic_experiment.py --model meta-llama/Llama-2-7b-hf
```

### Test Different Strengths
```bash
for s in 5 10 15 20; do
    python robust_demographic_experiment.py \
        --phase intervene \
        --intervention_strength $s
done
```

### Reduce Memory Usage
```bash
python robust_demographic_experiment.py --n_samples_per_category 25
```

### Use CPU
```bash
python robust_demographic_experiment.py --device cpu
```

## Output Files

```
robust_experiment_results/
├── extractions/
│   └── gender_attention_extractions.pkl      # Cached activations
└── intervention_results/
    ├── gender_attention_intervention_results.pkl  # Full results
    └── intervention_summary.json              # Key metrics ⭐
```

## Check Results

```bash
# Summary
cat robust_experiment_results/intervention_results/intervention_summary.json

# Pretty print
python -m json.tool robust_experiment_results/intervention_results/intervention_summary.json
```

## Example Scripts

```bash
cd examples
./quick_test.sh           # 15-minute test
./full_experiment.sh      # Full experiment
./compare_models.sh       # Model comparison
./sweep_parameters.sh     # Parameter search
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `--n_samples_per_category 25` |
| Slow extraction | `--device cuda` |
| Missing file error | Run `--phase extract` first |
| Import errors | Run from repo root |

## Expected Performance

| Configuration | Time | Memory | GPU |
|--------------|------|--------|-----|
| Quick test | 15 min | 4 GB | Optional |
| Single demo | 30 min | 8 GB | Recommended |
| Full experiment | 3-6 hours | 16 GB | Required |

## Interpreting Results

From `intervention_summary.json`:

```json
{
  "demographics": {
    "gender": {
      "baseline_accuracy": 0.62,      // Natural accuracy
      "intervention_accuracy": 0.71,  // With circuit steering
      "improvement": 9.0              // Percentage point gain ⭐
    }
  }
}
```

- **Positive improvement**: Circuit successfully encodes demographic
- **Near-zero improvement**: Circuit weakly encodes demographic
- **Negative improvement**: Circuit interference

## File Sizes

Typical extraction file sizes:
- Attention (100 samples/category): 1-3 GB per demographic
- MLP (100 samples/category): 500 MB - 1 GB per demographic

Reduce with `--n_samples_per_category 50`

## Key Demographics

Most reliable (well-represented in ANES):
- `gender` (binary: Man/Woman)
- `age` (3 categories)
- `race` (multiple categories)
- `education` (3 levels)
- `ideology` (3 categories)

## Key Political Questions

All 12 questions tested:
- `abortion`, `death_penalty`, `military_force`, `defense_spending`
- `govt_jobs`, `govt_help_blacks`, `colleges_opinion`, `dei_opinion`
- `journalist_access`, `transgender_bathrooms`
- `birthright_citizenship`, `immigration_policy`

## Advanced Usage

### Custom ANES Data
```bash
python robust_demographic_experiment.py \
    --anes_data_path /path/to/custom/anes.csv
```

### Skip Existing Extractions
```bash
python robust_demographic_experiment.py \
    --phase extract \
    --skip_existing
```

### Custom Output Directory
```bash
python robust_demographic_experiment.py \
    --output_dir my_experiment_results
```

### Different Random Seed
```bash
python robust_demographic_experiment.py --seed 123
```

## Getting Help

```bash
# Full help
python robust_demographic_experiment.py --help

# Documentation
cat README.md              # Overview
cat docs/QUICKSTART.md     # Tutorial
cat docs/TROUBLESHOOTING.md # Solutions
cat STRUCTURE.md           # Architecture
```

## Citation

```bibtex
@software{robust_demographic_circuits,
  title={Robust Demographic Circuit Interventions},
  year={2025},
  url={https://github.com/yourusername/RobustDemographicCircuits}
}
```
