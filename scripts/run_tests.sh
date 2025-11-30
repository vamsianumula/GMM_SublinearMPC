#!/bin/bash
set -e

# Set PYTHONPATH to include src
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "Running Existing Tests..."
mpirun -n 4 python3 src/mm_mpc/tests/test_batch1.py
mpirun -n 4 python3 src/mm_mpc/tests/test_batch2.py
mpirun -n 4 python3 src/mm_mpc/tests/test_batch3.py
mpirun -n 4 python3 src/mm_mpc/tests/test_batch4.py
mpirun -n 4 python3 src/mm_mpc/tests/test_batch5.py
mpirun -n 4 python3 src/mm_mpc/tests/test_io.py

echo "Running New Unit Tests..."
mpirun -n 4 python3 src/mm_mpc/tests/unit/test_vertex_state.py
mpirun -n 4 python3 src/mm_mpc/tests/unit/test_sublinearity.py
mpirun -n 4 python3 src/mm_mpc/tests/unit/test_finishing.py
mpirun -n 4 python3 src/mm_mpc/tests/unit/test_correctness_graphs.py

echo "Running Verification Script..."
mpirun -n 4 python3 scripts/verify_fixes.py

echo "ALL TESTS PASSED!"
