# Repository Structure

```
RobustDemographicCircuits/
│
├── README.md                           # Main documentation
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── setup.py                           # Package setup file
├── .gitignore                         # Git ignore patterns
├── CONTRIBUTING.md                     # Contribution guidelines
├── STRUCTURE.md                        # This file
│
├── robust_demographic_experiment.py    # Main experiment script
│
├── src/                               # Core modules
│   ├── probing_classifier.py          # Attention head probing
│   ├── baukit_activation_extraction.py # Activation extraction utilities
│   ├── anes_association_learning.py   # ANES data handling
│   ├── intervention_engine.py         # Attention intervention logic
│   └── mlp_intervention_engine.py     # MLP intervention logic
│
├── examples/                          # Example scripts
│   ├── quick_test.sh                  # Fast single demographic test
│   ├── full_experiment.sh             # Complete experiment
│   ├── compare_models.sh              # Model comparison
│   └── sweep_parameters.sh            # Parameter grid search
│
├── docs/                              # Documentation
│   ├── QUICKSTART.md                  # Quick start guide
│   └── TROUBLESHOOTING.md             # Common issues and solutions
│
├── data/                              # Data directory
│   └── README.md                      # Data download instructions
│
└── robust_experiment_results/         # Generated during experiments (gitignored)
    ├── extractions/                   # Cached activations
    │   ├── gender_attention_extractions.pkl
    │   └── ...
    └── intervention_results/          # Experiment results
        ├── gender_attention_intervention_results.pkl
        ├── intervention_summary.json
        └── ...
```

## File Descriptions

### Root Files

- **robust_demographic_experiment.py**: Main experimental framework
  - Command-line interface
  - Extraction phase (saves activations)
  - Intervention phase (k-fold validation)
  - Result aggregation and reporting

- **requirements.txt**: Python package dependencies
  - PyTorch, Transformers, scikit-learn
  - Baukit for activation extraction
  - Matplotlib/seaborn for visualization

- **setup.py**: Package installation configuration
  - Enables `pip install -e .`
  - Defines console scripts

### Source Modules (src/)

- **probing_classifier.py**: Linear probing implementation
  - Ridge regression classifiers
  - Cross-validation
  - Intervention weight extraction

- **baukit_activation_extraction.py**: Activation extraction
  - Attention head activations
  - MLP layer activations
  - Batch processing utilities

- **anes_association_learning.py**: Data handling
  - ANES data loading and preprocessing
  - Variable mappings
  - Synthetic data generation

- **intervention_engine.py**: Attention interventions
  - Hook-based activation steering
  - Multiple intervention strategies
  - Logit extraction for evaluation

- **mlp_intervention_engine.py**: MLP interventions
  - Layer-wise interventions
  - Similar interface to attention engine

### Examples (examples/)

All example scripts are executable bash scripts demonstrating common workflows:

- **quick_test.sh**: 15-minute test with single demographic
- **full_experiment.sh**: Multi-hour comprehensive experiment
- **compare_models.sh**: Test multiple model architectures
- **sweep_parameters.sh**: Grid search over intervention configs

### Documentation (docs/)

- **QUICKSTART.md**: Get running in 5 minutes
- **TROUBLESHOOTING.md**: Solutions to common problems

### Data (data/)

- **README.md**: Instructions for downloading ANES data
- Place `anes_timeseries_2024_csv_20250808.csv` here

## Workflow

### 1. Installation
```bash
git clone <repo>
cd RobustDemographicCircuits
pip install -r requirements.txt
```

### 2. Run Experiment
```bash
# Simple: run everything
python robust_demographic_experiment.py --demographics gender

# Advanced: two-phase workflow
python robust_demographic_experiment.py --phase extract
python robust_demographic_experiment.py --phase intervene
```

### 3. Analyze Results
```bash
cat robust_experiment_results/intervention_results/intervention_summary.json
```

## Key Design Principles

1. **Modularity**: Core functions in `src/`, main script coordinates
2. **Caching**: Extraction phase saves results to disk
3. **Flexibility**: All parameters configurable via CLI
4. **Reproducibility**: Fixed random seeds, deterministic splits
5. **Robustness**: K-fold validation prevents overfitting

## Adding New Features

### New Probe Type
1. Add extraction function in `baukit_activation_extraction.py`
2. Add intervention engine in new file (e.g., `new_intervention_engine.py`)
3. Update main script to handle new probe type

### New Demographic
1. Add to `ALL_DEMOGRAPHICS` in main script
2. Update prompt generation in `create_prompt_without_attribute()`
3. Ensure ANES data contains the demographic column

### New Political Question
1. Add to `ALL_POLITICAL_QUESTIONS` in main script
2. Add variable mapping in `anes_association_learning.py`
3. Ensure ANES data contains the question column

## Output Files

### Extraction Files (.pkl)
Binary pickle files containing:
- Activations tensor for each question
- Category labels
- Metadata (model, probe_type, timestamp)

### Intervention Results (.pkl)
Binary pickle files containing:
- Per-fold results (train/test splits, metrics)
- Per-question aggregates (mean/std across folds)
- Overall metrics
- Intervention weights for each fold

### Summary (.json)
Human-readable JSON with:
- Configuration parameters
- Per-demographic metrics
- Timestamp

## Memory Considerations

Typical memory usage:
- **Extraction**: 8-16GB (depends on n_samples_per_category)
- **Intervention**: 4-8GB (depends on top_k_heads)

Large extraction files:
- ~1-5GB per demographic (attention heads)
- ~500MB-2GB per demographic (MLP layers)

Use `--n_samples_per_category 50` to reduce memory usage.
