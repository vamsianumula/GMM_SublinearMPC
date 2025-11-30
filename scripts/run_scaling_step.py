#!/usr/bin/env python3
"""
scripts/run_scaling_step.py
Runs a single step of the scaling sweep.
Usage: mpirun -n <P> python3 scripts/run_scaling_step.py <P>
"""

import os
import sys
import argparse
from mpi4py import MPI

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mm_mpc.config import MPCConfig
from src.mm_mpc.driver import run_driver_with_io
from src.mm_mpc.utils import hashing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("p_ranks", type=int, help="Number of ranks (for directory naming)")
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_file = os.path.join(ROOT_DIR, "experiments", "large_dense.txt")
    results_dir = os.path.join(ROOT_DIR, "experiments", "results", "scaling", f"p{args.p_ranks}")
    
    if rank == 0:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            
    comm.Barrier()
    
    n = 1000
    p = 0.02
    m = int(n * (n-1) / 2 * p)
    
    config = MPCConfig(
        alpha=0.5,
        n_global=n,
        m_global=m,
        mem_per_cpu_gb=1.0,
        enable_metrics=True,
        enable_test_mode=False,
        metrics_output_dir=results_dir,
        S_edges=int(m/size * 5),
        R_rounds=10
    )
    
    hashing.init_seed(42)
    
    run_driver_with_io(comm, config, graph_file)

if __name__ == "__main__":
    main()
