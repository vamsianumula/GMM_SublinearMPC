#!/usr/bin/env python3
"""
scripts/run_single_trace.py
Runs a single trace experiment (4 ranks) and plots results.
Usage: mpirun -n 4 python3 scripts/run_single_trace.py
"""

import os
import sys
import shutil
import subprocess
from mpi4py import MPI

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mm_mpc.config import MPCConfig
from src.mm_mpc.driver import run_driver_with_io
from src.mm_mpc.utils import hashing

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_file = os.path.join(ROOT_DIR, "experiments", "large_dense.txt")
    results_dir = os.path.join(ROOT_DIR, "experiments", "results", "single_trace")
    
    # Generate graph if not exists (Rank 0 only)
    if rank == 0:
        if not os.path.exists(graph_file):
            print("Generating graph...")
            gen_script = os.path.join(ROOT_DIR, "scripts", "generate_graphs.py")
            subprocess.check_call([sys.executable, gen_script, "--type", "dense", "--n", "1000", "--p", "0.02", "--out", graph_file])
            
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        
    comm.Barrier()
    
    # Config
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
    
    if rank == 0:
        print(f"Experiment done. Results in {results_dir}")
        print(f"Run: python3 scripts/plot_metrics.py {os.path.join(results_dir, 'metrics_run.json')} --out {results_dir}")

if __name__ == "__main__":
    main()
