import numpy as np
from mpi4py import MPI
from mm_mpc.driver import run_driver_with_io
from mm_mpc.config import MPCConfig
from mm_mpc.utils import hashing
import os

def run_test_on_graph(comm, rank, edges, name, expected_size_min, expected_size_max=None):
    filename = f"test_graph_{name}.txt"
    if rank == 0:
        with open(filename, "w") as f:
            for u, v in edges:
                f.write(f"{u} {v}\n")
                
    comm.Barrier()
    
    # Calculate N and M
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    N = len(nodes) if nodes else 0
    M = len(edges)
    
    # Config
    # Use small S_edges to force distributed logic
    config = MPCConfig(alpha=0.5, n_global=N, m_global=M, 
                       S_edges=max(5, M // 2), R_rounds=2, mem_per_cpu_gb=1.0)
    
    try:
        matching = run_driver_with_io(comm, config, filename)
        
        if rank == 0:
            print(f"[{name}] Matching Size: {len(matching)}")
            
            # Verify Validity
            matched = set()
            for u, v in matching:
                assert u not in matched and v not in matched, f"[{name}] Vertex collision"
                matched.add(u)
                matched.add(v)
                
            # Verify Size
            assert len(matching) >= expected_size_min, f"[{name}] Matching too small: {len(matching)} < {expected_size_min}"
            if expected_size_max is not None:
                assert len(matching) <= expected_size_max, f"[{name}] Matching too large: {len(matching)} > {expected_size_max}"
                
            print(f"[{name}] PASSED")
            
    finally:
        if rank == 0 and os.path.exists(filename):
            os.remove(filename)
            
def test_cycle_graph():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hashing.init_seed(100)
    
    # Cycle C6: 0-1-2-3-4-5-0
    # Max matching = 3
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    run_test_on_graph(comm, rank, edges, "Cycle", 2, 3) # Greedy might get 2

def test_star_graph():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hashing.init_seed(101)
    
    # Star center 0, leaves 1..10
    # Max matching = 1
    edges = [(0, i) for i in range(1, 11)]
    run_test_on_graph(comm, rank, edges, "Star", 1, 1)

def test_complete_graph():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hashing.init_seed(102)
    
    # K6
    # Max matching = 3
    edges = []
    for i in range(6):
        for j in range(i+1, 6):
            edges.append((i, j))
    run_test_on_graph(comm, rank, edges, "Complete", 3, 3)

def test_bipartite_graph():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hashing.init_seed(103)
    
    # K3,3
    # Max matching = 3
    edges = []
    for i in range(3): # Left: 0,1,2
        for j in range(3, 6): # Right: 3,4,5
            edges.append((i, j))
    run_test_on_graph(comm, rank, edges, "Bipartite", 3, 3)

def test_disconnected_graph():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hashing.init_seed(104)
    
    # Two K3s: 0-1-2 and 3-4-5
    # Max matching = 1 + 1 = 2
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
    run_test_on_graph(comm, rank, edges, "Disconnected", 2, 2)

if __name__ == "__main__":
    test_cycle_graph()
    test_star_graph()
    test_complete_graph()
    test_bipartite_graph()
    test_disconnected_graph()
