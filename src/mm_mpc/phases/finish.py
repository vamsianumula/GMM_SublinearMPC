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
    config: MPCConfig
) -> list:
    rank = comm.Get_rank()
    
    # 1. Check Global Size
    active_indices = np.where(edge_state.active_mask)[0]
    local_count = len(active_indices)
    global_count = comm.allreduce(local_count, op=MPI.SUM)
    
    if global_count == 0:
        return []
        
    # Safety Threshold
    # If remaining edges > Threshold, gathering is unsafe.
    threshold = config.S_edges * config.small_threshold_factor
    
    if global_count > threshold:
        if rank == 0:
            print(f"[Finish] WARNING: Remaining edges ({global_active}) exceeds "
                  f"safety threshold ({threshold}). Skipping sequential finish.")
        return []

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