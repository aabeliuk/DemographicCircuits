#!/bin/bash
# Compare different LLM models on demographic circuit extraction
# Tests multiple model sizes

echo "Comparing demographic circuits across different model sizes..."

DEMOGRAPHICS="gender age race"
PROBE_TYPE="attention"
N_FOLDS=4

# Model 1: Llama-3.2-1B
echo ""
echo "Testing Llama-3.2-1B..."
python ../robust_demographic_experiment.py \
    --model meta-llama/Llama-3.2-1B \
    --probe_type $PROBE_TYPE \
    --demographics $DEMOGRAPHICS \
    --n_folds $N_FOLDS \
    --output_dir model_comparison/llama-1b

# Model 2: Llama-3.2-3B (if available)
echo ""
echo "Testing Llama-3.2-3B..."
python ../robust_demographic_experiment.py \
    --model meta-llama/Llama-3.2-3B \
    --probe_type $PROBE_TYPE \
    --demographics $DEMOGRAPHICS \
    --n_folds $N_FOLDS \
    --output_dir model_comparison/llama-3b

# You can add more models here
# Example:
# python ../robust_demographic_experiment.py \
#     --model meta-llama/Llama-2-7b-hf \
#     --probe_type $PROBE_TYPE \
#     --demographics $DEMOGRAPHICS \
#     --n_folds $N_FOLDS \
#     --output_dir model_comparison/llama2-7b

echo ""
echo "Model comparison complete!"
echo "Compare results in model_comparison/*/intervention_results/intervention_summary.json"
