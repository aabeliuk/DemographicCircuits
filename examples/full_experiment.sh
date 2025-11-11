#!/bin/bash
# Full experiment with all demographics
# This will take several hours to complete

echo "Starting full demographic circuit experiment..."
echo "This may take 3-6 hours depending on hardware"

# Phase 1: Extract activations for all demographics
echo ""
echo "Phase 1: Extracting activations..."
python ../robust_demographic_experiment.py \
    --phase extract \
    --model meta-llama/Llama-3.2-1B \
    --probe_type attention \
    --demographics gender age race education marital_status income ideology religion urban_rural \
    --n_samples_per_category 100 \
    --output_dir full_experiment_results

# Phase 2: Run interventions with k-fold validation
echo ""
echo "Phase 2: Running interventions with k-fold validation..."
python ../robust_demographic_experiment.py \
    --phase intervene \
    --model meta-llama/Llama-3.2-1B \
    --probe_type attention \
    --demographics gender age race education marital_status income ideology religion urban_rural \
    --n_folds 5 \
    --top_k_heads 10 \
    --intervention_strength 10.0 \
    --output_dir full_experiment_results

echo ""
echo "Full experiment complete!"
echo "Results saved to full_experiment_results/"
echo "See full_experiment_results/intervention_results/intervention_summary.json for summary"
