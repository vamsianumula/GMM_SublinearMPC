#!/bin/bash
set -e

# Configuration
N=10000
RESULTS_DIR="results_scaling"
mkdir -p $RESULTS_DIR

# Ranks to test
# Note: On a local machine, we are limited by cores.
# For this demo, we will run 2 and 4.
# On a cluster, you would extend this list: 2 4 8 16 32 64 ...
RANKS=(2 4)

echo "Starting Scalability Suite (N=$N)..."

for P in "${RANKS[@]}"; do
    echo "  Running with $P ranks..."
    OUT_FILE="$RESULTS_DIR/result_${P}.json"
    
    # Run the step script
    # Ensure PYTHONPATH is set
    export PYTHONPATH=$(pwd)/src
    
    mpirun -n $P python3 scripts/run_scaling_step.py --n $N --out $OUT_FILE
    
    echo "  -> Done. Saved to $OUT_FILE"
done

echo "All experiments completed."
echo "Generating plots..."
python3 scripts/plot_scaling.py --dir $RESULTS_DIR

echo "Scalability Suite Finished."
