"""
src/mm_mpc/phases/exponentiate.py
Phase 3: Graph Exponentiation.
Grows neighborhoods (balls) by R steps using sorted merging.
"""

import numpy as np
from mpi4py import MPI
from collections import defaultdict
from typing import Optional
from ..state_layout import EdgeState
from ..config import MPCConfig
from ..utils import mpi_helpers, hashing

def merge_sorted_unique(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Merges two arrays into a sorted unique array.
    
    PERFORMANCE NOTE: 
    np.union1d does a sort, which is O(N log N).
    Since inputs are already sorted, a custom C/Numba kernel could do this in O(N).
    For this Python reference, we accept the numpy overhead for stability.
    """
    return np.union1d(arr1, arr2)

def build_balls(
    comm: MPI.Comm,
    edge_state: EdgeState,
    config: MPCConfig,
    participating_mask: Optional[np.ndarray] = None
) -> None:
    """
    Grows the ball for participating edges.
    
    Args:
        participating_mask: Boolean array of edges in the current sparse subgraph H_s.
                            If None, defaults to all active, non-stalled edges.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 1. Determine Candidates
    # Must be Active AND Not Stalled AND (if provided) Participating in H_s
    candidates = edge_state.active_mask & ~edge_state.stalled
    if participating_mask is not None:
        candidates &= participating_mask
        
    active_indices = np.where(candidates)[0]
    
    # Current state of balls: dict mapping local_index -> np.array(eids)
    current_balls = {} 
    
    for idx in active_indices:
        eid = edge_state.edge_ids[idx]
        current_balls[idx] = np.array([eid], dtype=np.int64)
        
    # --- R-Round Loop ---
    for step in range(config.R_rounds):
        
        # --- Comm 1: Edge sends Ball to Vertices ---
        send_bufs = [[] for _ in range(size)]
        
        for idx, ball in current_balls.items():
            u, v = edge_state.edges_local[idx]
            eid = edge_state.edge_ids[idx]
            
            ball_len = len(ball)
            
            # Send to Owner(U)
            owner_u = hashing.get_vertex_owner(u, size)
            send_bufs[owner_u].extend([u, eid, ball_len])
            send_bufs[owner_u].extend(ball)
            
            # Send to Owner(V)
            owner_v = hashing.get_vertex_owner(v, size)
            send_bufs[owner_v].extend([v, eid, ball_len])
            send_bufs[owner_v].extend(ball)
            
        recv_data = mpi_helpers.exchange_buffers(comm, send_bufs, dtype=np.int64)
        
        # --- Vertex Aggregation ---
        v_inbox = defaultdict(list)
        v_subscribers = defaultdict(list)
        
        for r_buf in recv_data:
            cursor = 0
            n_buf = len(r_buf)
            while cursor < n_buf:
                target_v = r_buf[cursor]
                source_eid = r_buf[cursor+1]
                length = r_buf[cursor+2]
                cursor += 3
                
                ball_data = r_buf[cursor : cursor+length]
                cursor += length
                
                v_inbox[target_v].append(ball_data)
                v_subscribers[target_v].append(source_eid)
        
        # Compute "Super-Ball" per Vertex
        v_super_ball = {}
        for v, ball_list in v_inbox.items():
            if not ball_list:
                v_super_ball[v] = np.array([], dtype=np.int64)
            else:
                big_arr = np.concatenate(ball_list)
                v_super_ball[v] = np.unique(big_arr) 
                
        # --- Comm 2: Vertex sends Super-Ball back to Subscribers ---
        reply_bufs = [[] for _ in range(size)]
        
        for v, subscribers in v_subscribers.items():
            super_b = v_super_ball[v]
            sb_len = len(super_b)
            
            for eid in subscribers:
                owner_e = hashing.get_edge_owner_from_id(eid, size)
                reply_bufs[owner_e].extend([eid, sb_len])
                reply_bufs[owner_e].extend(super_b)
                
        recv_replies = mpi_helpers.exchange_buffers(comm, reply_bufs, dtype=np.int64)
        
        # --- Edge Update ---
        lookup = edge_state.id_to_index
        
        for r_buf in recv_replies:
            cursor = 0
            n_buf = len(r_buf)
            while cursor < n_buf:
                target_eid = r_buf[cursor]
                length = r_buf[cursor+1]
                cursor += 2
                
                incoming_ball = r_buf[cursor : cursor+length]
                cursor += length
                
                if target_eid in lookup:
                    idx = lookup[target_eid]
                    
                    # Merge and Sort
                    new_ball = merge_sorted_unique(current_balls[idx], incoming_ball)
                    
                    # MEMORY GUARD
                    if len(new_ball) > config.S_edges:
                        raise MemoryError(
                            f"Rank {rank}: Edge {target_eid} ball size {len(new_ball)} "
                            f"exceeded limit {config.S_edges} in step {step}."
                        )
                    
                    current_balls[idx] = new_ball

    # --- Finalize State ---
    # We must clear old ball storage to prevent index misalignment
    m = len(edge_state.edges_local)
    edge_state.ball_offsets[:] = 0
    
    lengths = np.zeros(m, dtype=np.int64)
    for idx, ball in current_balls.items():
        lengths[idx] = len(ball)
        
    np.cumsum(lengths, out=edge_state.ball_offsets[1:])
    
    total_items = edge_state.ball_offsets[-1]
    edge_state.ball_storage = np.zeros(total_items, dtype=np.int64)
    
    for idx, ball in current_balls.items():
        start = edge_state.ball_offsets[idx]
        end = start + len(ball)
        edge_state.ball_storage[start:end] = ball