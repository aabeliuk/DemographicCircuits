#!/bin/bash
# Quick test with single demographic
# Useful for testing setup and getting quick results

echo "Running quick test with gender demographic..."

python ../robust_demographic_experiment.py \
    --model meta-llama/Llama-3.2-1B \
    --probe_type attention \
    --demographics gender \
    --n_folds 2 \
    --n_samples_per_category 50 \
    --top_k_heads 5 \
    --intervention_strength 10.0 \
    --output_dir quick_test_results

echo "Quick test complete! Results in quick_test_results/"
