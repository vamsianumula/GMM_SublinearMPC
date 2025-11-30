#!/bin/bash
# scripts/run_local.sh
# Usage: ./scripts/run_local.sh <num_ranks> <input_file>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num_ranks> <input_file>"
    exit 1
fi

NP=$1
INPUT=$2

# 1. Set PYTHONPATH to include the src directory
# This allows 'import mm_mpc' to work without installation
# export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# 2. Run with default parameters for local testing
# Alpha 0.5 is good for testing sparsification logic on small graphs
echo "--- Running MPC Maximal Matching on Local Machine ---"
echo "Ranks: $NP"
echo "Input: $INPUT"


PYTHONPATH=src mpirun -n $NP python3 -m mm_mpc.cli \
    --input "$INPUT" \
    --n 1000 \
    --m 5000 \
    --alpha 0.5 \
    --mem 1.0