"""
src/mm_mpc/phases/finish.py
Phase 6: Finish Small Components.
Includes safety check for Rank 0 capacity.
"""

import numpy as np
from mpi4py import MPI
from ..state_layout import EdgeState
from ..config import MPCConfig

def solve_sequential_greedy(edges: np.ndarray) -> list:
    matching = []
    matched_verts = set()
    for u, v in edges:
        if u not in matched_verts and v not in matched_verts:
            matched_verts.add(u)
            matched_verts.add(v)
            matching.append((u, v))
    return matching

def finish_small_components(
    comm: MPI.Comm,
    edge_state: EdgeState,
    vertex_state: EdgeState,
    config: MPCConfig
) -> list:
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 1. Check Global Size
    active_indices = np.where(edge_state.active_mask)[0]
    local_count = len(active_indices)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    
    if global_count == 0:
        return []
        
    # Safety Threshold
    threshold = config.S_edges * config.small_threshold_factor
    
    if global_count > threshold:
        if rank == 0:
            print(f"[Finish] WARNING: Remaining edges ({global_count}) exceeds "
                  f"safety threshold ({threshold}). Switching to Distributed Finishing.")
        
        # Distributed Fallback Loop
        from . import exponentiate, local_mis, integrate
        
        extra_matches = []
        # Run 5 rounds of standard MIS on the remaining graph
        # We temporarily set R=1 in config for this? Or just pass R=1?
        # Exponentiate uses config.R_rounds. We can temporarily patch it or create a temp config.
        
        # Create a temp config with R=1
        from dataclasses import replace
        temp_config = replace(config, R_rounds=1)
        
        for i in range(5):
            if rank == 0: print(f"[Finish] Distributed Round {i+1}")
            
            # 1. Build 1-hop balls (on ALL active edges, no sparsification)
            # We pass participating_mask=None (implies all active & non-stalled)
            # But we should ensure nothing is stalled?
            # Stall logic might have left some edges stalled. We should unstall them?
            # Yes, for finishing, we want to process everything.
            edge_state.stalled[:] = False
            
            try:
                exponentiate.build_balls(comm, edge_state, vertex_state, temp_config, participating_mask=None)
            except MemoryError:
                if rank == 0: print("[Finish] OOM during fallback. Aborting finish.")
                break
                
            # 2. MIS
            chosen = local_mis.run_greedy_mis(edge_state, phase=999+i, participating_mask=None)
            
            # 3. Integrate
            new_m = integrate.update_matching_and_prune(comm, edge_state, vertex_state, chosen, size)
            extra_matches.extend(new_m)
            
            # Check if done
            n_active = np.sum(edge_state.active_mask)
            g_active = comm.allreduce(n_active, op=MPI.SUM)
            if g_active == 0:
                break
                
        return extra_matches

    # 2. Gather active edges
    my_edges = edge_state.edges_local[active_indices]
    
    send_data = my_edges.flatten().astype(np.int64)
    send_count = np.array([len(send_data)], dtype=np.int32)
    recv_counts = np.zeros(comm.Get_size(), dtype=np.int32)
    
    comm.Gather(send_count, recv_counts, root=0)
    
    recv_buf = None
    if rank == 0:
        recv_buf = np.empty(np.sum(recv_counts), dtype=np.int64)
    
    displs = None
    if rank == 0:
        displs = np.concatenate(([0], np.cumsum(recv_counts)[:-1]))
        
    comm.Gatherv([send_data, MPI.INT64_T], [recv_buf, recv_counts, displs, MPI.INT64_T], root=0)
    
    # 3. Solve Locally
    extra_matches = []
    if rank == 0:
        edges = recv_buf.reshape((-1, 2))
        extra_matches = solve_sequential_greedy(edges)
                
    return extra_matches