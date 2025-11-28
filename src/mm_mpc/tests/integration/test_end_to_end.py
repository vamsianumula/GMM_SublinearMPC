"""
tests/integration/test_end_to_end.py
Standard integration test for CI/CD.
"""
import os
import shutil
import pytest
from mpi4py import MPI
from mm_mpc.driver import run_driver_with_io
from mm_mpc.config import MPCConfig
from mm_mpc.utils import hashing

def test_path_graph_matching():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Path Graph 0-1-2-3
    filename = "test_data_path.txt"
    if rank == 0:
        os.makedirs("test_data", exist_ok=True)
        with open(filename, "w") as f:
            f.write("0 1\n1 2\n2 3\n")
            
    comm.Barrier()
    hashing.init_seed(999)
    
    # Configuration
    config = MPCConfig(
        alpha=0.5, n_global=4, m_global=3,
        S_edges=100, R_rounds=2, mem_per_cpu_gb=1.0
    )
    
    try:
        run_driver_with_io(comm, config, filename)
        # Note: We rely on the driver's internal assertions and exit code 0
        # for success in this context.
    except Exception as e:
        pytest.fail(f"End-to-End failed: {e}")
    finally:
        if rank == 0 and os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_path_graph_matching()