"""
tests/scale/test_correctness_suite.py
Systematic correctness testing for MPC Maximal Matching.
Generates synthetic graphs, runs MPC, and verifies Maximality properties.
"""

import os
import random
import numpy as np
import pytest
from mpi4py import MPI
from mm_mpc.driver import run_driver_with_io
from mm_mpc.config import MPCConfig
from mm_mpc.utils import hashing

# --- Graph Generators ---

def generate_random_graph(filename, n, p_edge):
    """Erdos-Renyi G(n, p) graph."""
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p_edge:
                edges.append((u, v))
    
    with open(filename, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")
    return edges

def generate_star_graph(filename, n):
    """
    Star graph: Center 0 connected to 1..n-1.
    Tests high-degree stalling logic.
    """
    edges = []
    for i in range(1, n):
        edges.append((0, i))
        
    with open(filename, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")
    return edges

def generate_dense_clique(filename, n):
    """Complete graph K_n."""
    return generate_random_graph(filename, n, 1.0)

# --- Verifier ---

def verify_maximal_matching(all_edges, matching):
    """
    Checks two properties:
    1. Validity: No two edges in matching share a vertex.
    2. Maximality: Every edge in G has at least one endpoint matched.
    """
    matched_verts = set()
    
    # 1. Check Validity
    for u, v in matching:
        if u in matched_verts:
            return False, f"Vertex {u} matched twice!"
        if v in matched_verts:
            return False, f"Vertex {v} matched twice!"
        matched_verts.add(u)
        matched_verts.add(v)
        
    # 2. Check Maximality
    # If M is maximal, then for every edge (u, v) in G,
    # either u is matched OR v is matched.
    for u, v in all_edges:
        if u not in matched_verts and v not in matched_verts:
            return False, f"Edge ({u}, {v}) is not covered! Matching is not maximal."
            
    return True, "OK"

# --- Test Runner ---

def run_parametric_test(graph_type, n, p_val=0.1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    filename = f"test_graph_{graph_type}_{n}.txt"
    all_edges = []
    
    # 1. Generate Data (Rank 0 only)
    if rank == 0:
        print(f"\n[Test] Generating {graph_type} (n={n})...")
        if graph_type == "random":
            all_edges = generate_random_graph(filename, n, p_val)
        elif graph_type == "star":
            all_edges = generate_star_graph(filename, n)
        elif graph_type == "clique":
            all_edges = generate_dense_clique(filename, n)
            
    comm.Barrier()
    
    # 2. Setup Config
    # We use aggressive parameters to force the algorithm to work hard
    # S_edges small ensures we test distributed logic
    hashing.init_seed(42 + n)
    config = MPCConfig(
        alpha=0.5, 
        n_global=n, 
        m_global=n*n, # upper bound
        S_edges=max(n // 2, 50), # Force splitting across ranks
        R_rounds=2, 
        mem_per_cpu_gb=1.0
    )
    
    # 3. Run
    try:
        matching = run_driver_with_io(comm, config, filename)
    except Exception as e:
        if rank == 0:
            print(f"CRASH in {graph_type}: {e}")
        raise e
    finally:
        # Cleanup
        if rank == 0 and os.path.exists(filename):
            os.remove(filename)
            
    # 4. Verify (Rank 0 only)
    if rank == 0:
        is_valid, msg = verify_maximal_matching(all_edges, matching)
        if not is_valid:
            pytest.fail(f"Verification Failed for {graph_type}: {msg}")
        else:
            print(f"SUCCESS: {graph_type} (n={n}) -> Size {len(matching)}")

# --- Pytest Entry Points ---

def test_random_graph_small():
    run_parametric_test("random", 50, 0.3)

def test_star_graph():
    # Star graph is tricky because one node has degree N-1.
    # This stresses the Stalling logic.
    run_parametric_test("star", 100)

def test_clique_small():
    # Clique requires many rounds to resolve conflicts
    run_parametric_test("clique", 30)

if __name__ == "__main__":
    # Manual run support
    test_random_graph_small()
    test_star_graph()
    test_clique_small()