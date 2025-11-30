import sys
import os
import numpy as np
from mpi4py import MPI
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from mm_mpc.cli import main as cli_main
from mm_mpc.config import MPCConfig
from mm_mpc.driver import run_driver_with_io

def generate_random_graph(n, m, path):
    edges = set()
    while len(edges) < m:
        u = np.random.randint(0, n)
        v = np.random.randint(0, n)
        if u != v:
            u, v = min(u, v), max(u, v)
            edges.add((u, v))
            
    with open(path, 'w') as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")
            
def verify_matching(matching, input_path):
    # Load original graph
    adj = {}
    with open(input_path, 'r') as f:
        for line in f:
            u, v = map(int, line.split())
            if u not in adj: adj[u] = []
            if v not in adj: adj[v] = []
            adj[u].append(v)
            adj[v].append(u)
            
    matched_verts = set()
    for u, v in matching:
        if u in matched_verts or v in matched_verts:
            return False, f"Vertex in multiple edges: {u} or {v}"
        matched_verts.add(u)
        matched_verts.add(v)
        
    # Check maximality (simple check)
    # Iterate all edges, check if at least one endpoint is matched
    with open(input_path, 'r') as f:
        for line in f:
            u, v = map(int, line.split())
            if u not in matched_verts and v not in matched_verts:
                return False, f"Edge {u}-{v} is not covered"
                
    return True, "OK"

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Create temp graph
    tmp_file = "test_graph.txt"
    if rank == 0:
        generate_random_graph(100, 500, tmp_file)
    comm.Barrier()
    
    # Run Driver
    # We mock args
    class Args:
        input = tmp_file
        n_global = 100
        m_global = 500
        alpha = 0.5
        mem_per_cpu = 1.0
        
    config = MPCConfig.from_args(Args(), comm.Get_size())
    
    try:
        matching = run_driver_with_io(comm, config, tmp_file)
        
        if rank == 0:
            valid, msg = verify_matching(matching, tmp_file)
            if valid:
                print("SUCCESS: Matching is valid and maximal.")
            else:
                print(f"FAILURE: {msg}")
                sys.exit(1)
                
    except Exception as e:
        print(f"Rank {rank} failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if rank == 0 and os.path.exists(tmp_file):
            os.remove(tmp_file)
