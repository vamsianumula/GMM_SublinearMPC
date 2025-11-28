"""
src/mm_mpc/phases/finish.py
Phase 6: Finish Small Components (Gather to Rank 0).
"""

import numpy as np
from mpi4py import MPI
from ..state_layout import EdgeState
from ..config import MPCConfig

def finish_small_components(
    comm: MPI.Comm,
    edge_state: EdgeState,
    config: MPCConfig
) -> list:
    rank = comm.Get_rank()
    
    # 1. Gather active edges
    active_indices = np.where(edge_state.active_mask)[0]
    my_edges = edge_state.edges_local[active_indices] # (k, 2)
    
    # Flatten
    send_data = my_edges.flatten().astype(np.int64)
    send_count = np.array([len(send_data)], dtype=np.int32)
    
    # Gather counts
    recv_counts = np.zeros(comm.Get_size(), dtype=np.int32)
    comm.Gather(send_count, recv_counts, root=0)
    
    # Gather data
    recv_buf = None
    if rank == 0:
        recv_buf = np.empty(np.sum(recv_counts), dtype=np.int64)
    
    displs = None
    if rank == 0:
        displs = np.concatenate(([0], np.cumsum(recv_counts)[:-1]))
        
    comm.Gatherv([send_data, MPI.INT64_T], [recv_buf, recv_counts, displs, MPI.INT64_T], root=0)
    
    # 2. Solve Locally
    extra_matches = []
    if rank == 0:
        edges = recv_buf.reshape((-1, 2))
        # Greedy sequential
        matched = set()
        for u, v in edges:
            if u not in matched and v not in matched:
                matched.add(u)
                matched.add(v)
                extra_matches.append((u, v))
                
    return extra_matches