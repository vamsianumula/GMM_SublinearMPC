import os
import pytest
from mpi4py import MPI
from mm_mpc.driver import run_driver_with_io
from mm_mpc.config import MPCConfig
from mm_mpc.utils import hashing

def test_full_pipeline():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    filename = "test_graph_batch5.txt"
    if rank == 0:
        with open(filename, "w") as f:
            f.write("0 1\n1 2\n2 3\n")
            
    comm.Barrier()
    hashing.init_seed(100)
    config = MPCConfig(alpha=0.5, n_global=4, m_global=3, S_edges=100, R_rounds=2, mem_per_cpu_gb=1.0)
    
    try:
        run_driver_with_io(comm, config, filename)
    except Exception as e:
        print(f"Rank {rank} ERROR: {e}")
        comm.Abort(1)
    finally:
        if rank == 0 and os.path.exists(filename): os.remove(filename)

if __name__ == "__main__":
    test_full_pipeline()