"""
tests/test_batch3.py
Verifies Sparsification (Degree Calculation) and Stalling.
"""

import numpy as np
from mpi4py import MPI
from mm_mpc.state_layout import init_edge_state
from mm_mpc.phases import sparsify, stall
from mm_mpc.config import MPCConfig
from mm_mpc.utils import hashing

def test_sparsification_logic():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Init seed
    hashing.init_seed(42)
    
    # 1. Setup a Triangle Graph (Cycle C3)
    # Edge owner logic MUST match sparsify.py's expectations
    raw_edges = [(0, 1), (1, 2), (2, 0)]
    
    local_edges = []
    local_ids = []
    
    for u, v in raw_edges:
        # Use the consistent ownership function
        owner = hashing.get_edge_owner(u, v, size)
        
        if owner == rank:
            local_edges.append([u, v])
            eid = hashing.get_edge_id(u, v)
            local_ids.append(eid)
            
    local_edges = np.array(local_edges, dtype=np.int64)
    local_ids = np.array(local_ids, dtype=np.int64)
    
    if len(local_edges) == 0:
        local_edges = np.empty((0, 2), dtype=np.int64)
        
    # Init State
    state = init_edge_state(local_edges, local_ids)
    
    # Initialize VertexState
    from mm_mpc.state_layout import init_vertex_state
    vertex_state = init_vertex_state(comm, state)
    
    # 2. Run Sparsification (p=1.1 -> All participate)
    participating = sparsify.compute_phase_participation(state, phase=1, iteration=0, p_val=1.1)
    
    # 3. Compute Degrees
    sparsify.compute_deg_in_sparse(comm, state, vertex_state, participating, size)
    
    # 4. Verify Degrees
    # Triangle: deg_L = 2
    if len(state.deg_in_sparse) > 0:
        print(f"Rank {rank} Degrees: {state.deg_in_sparse}")
        assert np.all(state.deg_in_sparse == 2), f"Rank {rank}: Expected 2, got {state.deg_in_sparse}"
        
    comm.Barrier()
    if rank == 0:
        print("Sparsification Degree Test: PASSED")

def test_stalling_logic():
    config = MPCConfig(alpha=0.5, n_global=100, m_global=100, 
                       S_edges=100, R_rounds=2, mem_per_cpu_gb=1.0)
    
    # T ~ 100^(1/2) = 10
    degs = np.array([5, 15, 2], dtype=np.int32)
    ids = np.array([100, 200, 300], dtype=np.int64)
    edges = np.zeros((3, 2), dtype=np.int64)
    
    state = init_edge_state(edges, ids)
    state.deg_in_sparse = degs
    
    stall.apply_stalling(state, phase=1, config=config)
    
    assert state.stalled[0] == False
    assert state.stalled[1] == True
    assert state.stalled[2] == False
    
    print("Stalling Test: PASSED")

if __name__ == "__main__":
    test_sparsification_logic()
    test_stalling_logic()