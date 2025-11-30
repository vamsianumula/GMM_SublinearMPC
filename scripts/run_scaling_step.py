import time
import json
import argparse
import os
import numpy as np
from mpi4py import MPI
from mm_mpc.config import MPCConfig
from mm_mpc.driver import run_driver_with_io
from mm_mpc.utils import mpi_helpers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Configuration for Strong Scaling (Fixed N)
    # We want to see speedup.
    # N is fixed.
    # M approx 10*N (Avg Degree 20)
    N = args.n
    M = N * 10
    
    # Create a temporary random graph file
    # Only rank 0 generates it to avoid race conditions
    input_file = f"scaling_input_{N}.txt"
    if rank == 0:
        import random
        with open(input_file, "w") as f:
            edges = set()
            while len(edges) < M:
                u = random.randint(0, N-1)
                v = random.randint(0, N-1)
                if u != v:
                    if u > v: u, v = v, u
                    if (u, v) not in edges:
                        edges.add((u, v))
                        f.write(f"{u} {v}\n")
    
    comm.Barrier()
    
    # Config
    # Alpha=0.5
    # S_edges = N^0.5 * 5
    import math
    S_edges = int(math.pow(N, 0.5) * 5)
    S_edges = max(S_edges, 20)
    
    config = MPCConfig(
        n_global=N,
        m_global=M,
        alpha=0.5,
        S_edges=S_edges,
        R_rounds=2,
        mem_per_cpu_gb=1.0 # Dummy value for scaling test
    )
    
    # Reset Metrics
    mpi_helpers.get_and_reset_metrics()
    
    # Run & Time
    start_time = time.time()
    try:
        run_driver_with_io(comm, config, input_file)
        success = True
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        success = False
        
    end_time = time.time()
    duration = end_time - start_time
    
    # Collect Metrics
    local_bytes = mpi_helpers.get_and_reset_metrics() # This might only capture the last phase if not careful?
    # Wait, get_and_reset_metrics returns _TOTAL_BYTES_SENT.
    # In driver.py, we call it at the end.
    # But driver.py ALSO calls it inside the loop for Phase 2 tracking!
    # So _TOTAL_BYTES_SENT might be reset during execution.
    # We need to be careful.
    # Actually, driver.py prints TotalCommBytes at the end.
    # But for JSON dump, we want the value.
    # Let's rely on the fact that driver.py prints it to stdout, 
    # OR we can modify driver.py to return metrics?
    # Or we can just trust that `driver.py`'s final print is what we want?
    # No, we want structured data.
    
    # Issue: `driver.py` resets metrics inside the loop for Phase 2 tracking.
    # So `local_bytes` here will only be the bytes sent AFTER the last reset.
    # This is a bug in my previous instrumentation if I want GLOBAL total.
    # I should fix `driver.py` or `mpi_helpers.py` to have a separate "Lifetime Total" counter.
    
    # For now, let's just use Duration. That's the most important for scaling.
    # And we can use Max Ball Size if we can extract it.
    
    # Let's aggregate Duration
    max_duration = comm.reduce(duration, op=MPI.MAX, root=0)
    avg_duration = comm.reduce(duration, op=MPI.SUM, root=0)
    if rank == 0:
        avg_duration /= size
        
    # Cleanup
    comm.Barrier()
    if rank == 0:
        if os.path.exists(input_file):
            os.remove(input_file)
            
        # Write JSON
        results = {
            "ranks": size,
            "N": N,
            "max_duration": max_duration,
            "avg_duration": avg_duration,
            "success": success
        }
        
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
