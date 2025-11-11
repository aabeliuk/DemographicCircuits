#!/bin/bash
# Parameter sweep: Test different intervention configurations
# Assumes extraction phase is already complete

echo "Running parameter sweep for intervention configurations..."

DEMOGRAPHIC="gender"
MODEL="meta-llama/Llama-3.2-1B"
PROBE_TYPE="attention"

# First, extract activations (run once)
echo "Step 1: Extracting activations (if not already done)..."
python ../robust_demographic_experiment.py \
    --phase extract \
    --model $MODEL \
    --probe_type $PROBE_TYPE \
    --demographics $DEMOGRAPHIC \
    --output_dir param_sweep \
    --skip_existing

# Sweep intervention strength
echo ""
echo "Step 2: Sweeping intervention strength..."
for strength in 5.0 10.0 15.0 20.0; do
    echo "  Testing intervention_strength=$strength"
    python ../robust_demographic_experiment.py \
        --phase intervene \
        --model $MODEL \
        --probe_type $PROBE_TYPE \
        --demographics $DEMOGRAPHIC \
        --intervention_strength $strength \
        --top_k_heads 10 \
        --n_folds 4 \
        --output_dir param_sweep/strength_${strength}
done

# Sweep top-k heads
echo ""
echo "Step 3: Sweeping top-k heads..."
for k in 5 10 15 20; do
    echo "  Testing top_k_heads=$k"
    python ../robust_demographic_experiment.py \
        --phase intervene \
        --model $MODEL \
        --probe_type $PROBE_TYPE \
        --demographics $DEMOGRAPHIC \
        --intervention_strength 10.0 \
        --top_k_heads $k \
        --n_folds 4 \
        --output_dir param_sweep/topk_${k}
done

echo ""
echo "Parameter sweep complete!"
echo "Results saved to param_sweep/"
echo ""
echo "To analyze results, compare the intervention_summary.json files:"
echo "  cat param_sweep/strength_*/intervention_results/intervention_summary.json"
echo "  cat param_sweep/topk_*/intervention_results/intervention_summary.json"
