"""
tests/test_batch4.py
Verifies Exponentiation and Local MIS.
"""

import numpy as np
from mpi4py import MPI
from mm_mpc.state_layout import init_edge_state
from mm_mpc.phases import exponentiate, local_mis
from mm_mpc.config import MPCConfig
from mm_mpc.utils import hashing

def test_exponentiation_and_mis():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    hashing.init_seed(42)
    
    # 1. Setup Path Graph 0-1-2-3
    raw_edges = [(0, 1), (1, 2), (2, 3)]
    local_edges = []
    local_ids = []
    
    for u, v in raw_edges:
        eid = hashing.get_edge_id(u, v)
        owner = hashing.get_edge_owner_from_id(eid, size)
        if owner == rank:
            local_edges.append([u, v])
            local_ids.append(eid)
            
    local_edges = np.array(local_edges, dtype=np.int64)
    local_ids = np.array(local_ids, dtype=np.int64)
    
    if len(local_edges) == 0:
        local_edges = np.empty((0, 2), dtype=np.int64)
        
    state = init_edge_state(local_edges, local_ids)
    
    # Initialize VertexState
    from mm_mpc.state_layout import init_vertex_state
    vertex_state = init_vertex_state(comm, state)
    
    # 2. Run Exponentiation
    config = MPCConfig(alpha=0.5, n_global=4, m_global=3, 
                       S_edges=100, R_rounds=1, mem_per_cpu_gb=1.0)
    
    # Simulate that all edges participate in this phase
    all_participating = np.ones(len(local_edges), dtype=bool)
    
    exponentiate.build_balls(comm, state, vertex_state, config, participating_mask=all_participating)
    
    # Verify Ball Contents for Edge (1,2)
    target_eid = hashing.get_edge_id(1, 2)
    if target_eid in state.id_to_index:
        idx = state.id_to_index[target_eid]
        start = state.ball_offsets[idx]
        end = state.ball_offsets[idx+1]
        ball = state.ball_storage[start:end]
        
        print(f"Rank {rank}: Ball for (1,2) has size {len(ball)}")
        eid_01 = hashing.get_edge_id(0, 1)
        eid_23 = hashing.get_edge_id(2, 3)
        assert target_eid in ball
        assert eid_01 in ball
        assert eid_23 in ball
        
    comm.Barrier()
    if rank == 0:
        print("Exponentiation Test: PASSED")
        
    # 3. Run MIS
    chosen = local_mis.run_greedy_mis(state, phase=1)
    
    local_chosen_edges = state.edges_local[chosen]
    all_chosen = comm.gather(local_chosen_edges, root=0)
    
    if rank == 0:
        # Robust handling for empty results
        non_empty = [a for a in all_chosen if a.size > 0]
        
        if not non_empty:
            print("No edges chosen (Possible if all priorities low? Unlikely active graph)")
            # In a path graph, MIS must have size >= 1
            assert False, "MIS returned empty matching on path graph!"
        else:
            flat_chosen = np.vstack(non_empty)
            print(f"Chosen Edges:\n{flat_chosen}")
            
            # Verify Independent Set property
            v_counts = {}
            for u, v in flat_chosen:
                v_counts[u] = v_counts.get(u, 0) + 1
                v_counts[v] = v_counts.get(v, 0) + 1
                
            for v, c in v_counts.items():
                assert c <= 1, f"Vertex {v} matched {c} times! Not a matching."
            
            print("MIS Test: PASSED")

if __name__ == "__main__":
    test_exponentiation_and_mis()