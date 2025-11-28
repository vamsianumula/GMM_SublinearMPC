"""
src/mm_mpc/cli.py
Command Line Interface.
"""

import sys
import argparse
from mpi4py import MPI
from .config import MPCConfig
from .driver import run_driver_with_io
from .utils import hashing

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Edge list file")
    parser.add_argument("--n", type=int, required=True, help="Num vertices")
    parser.add_argument("--m", type=int, required=True, help="Num edges")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--mem", type=float, default=1.0)
    
    args = parser.parse_args()
    
    config = MPCConfig.from_args(
        type('Args', (object,), {
            'alpha': args.alpha,
            'n_global': args.n,
            'm_global': args.m,
            'mem_per_cpu': args.mem
        }), size
    )
    
    hashing.init_seed(42)
    
    run_driver_with_io(comm, config, args.input)

if __name__ == "__main__":
    main()