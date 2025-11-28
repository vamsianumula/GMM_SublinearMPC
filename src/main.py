import sys
import argparse
import traceback
from mpi4py import MPI
from mm_mpc.config import MPCConfig
from mm_mpc.driver import run_driver_with_io
from mm_mpc.utils import hashing

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True)
        parser.add_argument("--n", type=int, default=100)
        parser.add_argument("--m", type=int, default=100)
        parser.add_argument("--alpha", type=float, default=0.2)
        parser.add_argument("--mem", type=float, default=1.0)
        args = parser.parse_args()
        
        config = MPCConfig.from_args(
            type('A', (object,), {'alpha': args.alpha, 'n_global': args.n, 'm_global': args.m, 'mem_per_cpu': args.mem}), 
            comm.Get_size()
        )
        
        hashing.init_seed(42)
        run_driver_with_io(comm, config, args.input)
        
    except Exception as e:
        print(f"Rank {rank} CRASHED: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        comm.Abort(1) # CRITICAL: Kills all ranks

if __name__ == "__main__":
    main()