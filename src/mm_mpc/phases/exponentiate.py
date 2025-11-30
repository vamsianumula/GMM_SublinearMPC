"""
src/mm_mpc/phases/exponentiate.py
Phase 3: Graph Exponentiation.
Grows neighborhoods (balls) by R steps using linear-time merging.
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
    Merges two sorted unique arrays into a sorted unique array.
    Currently uses np.union1d (O(N log N)). 
    In production C++/Cython, this would be a linear O(N) scan.
    """
    return np.union1d(arr1, arr2)

def build_balls(
    comm: MPI.Comm, 
    edge_state: EdgeState, 
    config: MPCConfig, 
    participating_mask: Optional[np.ndarray] = None,
    tracker=None
):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 1. Determine Candidates
    candidates = edge_state.active_mask & ~edge_state.stalled
    if participating_mask is not None:
        candidates &= participating_mask
    active_indices = np.where(candidates)[0]
    
    # Current state of balls: dict mapping local_index -> np.array(eids)
    current_balls = {idx: np.array([edge_state.edge_ids[idx]], dtype=np.int64) for idx in active_indices}
    
    # --- R-Round Loop ---
    for step in range(config.R_rounds):
        # 1. Send to Vertices
        send_bufs = [[] for _ in range(size)]
        for idx, ball in current_balls.items():
            u, v = edge_state.edges_local[idx]
            eid = edge_state.edge_ids[idx]
            blen = len(ball)
            
            for target in [u, v]:
                owner = hashing.get_vertex_owner(target, size)
                send_bufs[owner].extend([target, eid, blen])
                send_bufs[owner].extend(ball)
                
        recv_data = mpi_helpers.exchange_buffers(comm, send_bufs, dtype=np.int64, tracker=tracker)
        
        # 2. Vertex Aggregation
        v_inbox = defaultdict(list)
        v_subscribers = defaultdict(list)
        for r_buf in recv_data:
            cursor = 0
            n = len(r_buf)
            while cursor < n:
                tv, seid, length = r_buf[cursor], r_buf[cursor+1], r_buf[cursor+2]
                cursor += 3
                v_inbox[tv].append(r_buf[cursor : cursor+length])
                v_subscribers[tv].append(seid)
                cursor += length
                
        # 3. Reply to Edges
        reply_bufs = [[] for _ in range(size)]
        for v, balls in v_inbox.items():
            if not balls: 
                super_b = np.array([], dtype=np.int64)
            else: 
                # Optimization: concat then unique is faster than iterative union
                super_b = np.unique(np.concatenate(balls))
            
            sblen = len(super_b)
            for eid in v_subscribers[v]:
                dest = hashing.get_edge_owner_from_id(eid, size)
                reply_bufs[dest].extend([eid, sblen])
                reply_bufs[dest].extend(super_b)
                
        recv_replies = mpi_helpers.exchange_buffers(comm, reply_bufs, dtype=np.int64, tracker=tracker)
        
        # 4. Merge Updates
        lookup = edge_state.id_to_index
        for r_buf in recv_replies:
            cursor = 0
            n = len(r_buf)
            while cursor < n:
                teid, length = r_buf[cursor], r_buf[cursor+1]
                cursor += 2
                inc = r_buf[cursor : cursor+length]
                cursor += length
                
                if teid in lookup:
                    idx = lookup[teid]
                    # Use the helper function here
                    new_ball = merge_sorted_unique(current_balls[idx], inc)
                    
                    if len(new_ball) > config.S_edges:
                        raise MemoryError(f"Rank {rank}: Ball size {len(new_ball)} exceeded {config.S_edges}")
                    current_balls[idx] = new_ball

    # --- Finalize State ---
    m = len(edge_state.edges_local)
    edge_state.ball_offsets[:] = 0
    lengths = np.zeros(m, dtype=np.int64)
    for idx, ball in current_balls.items(): 
        lengths[idx] = len(ball)
    np.cumsum(lengths, out=edge_state.ball_offsets[1:])
    
    total = edge_state.ball_offsets[-1]
    edge_state.ball_storage = np.zeros(total, dtype=np.int64)
    for idx, ball in current_balls.items():
        start = edge_state.ball_offsets[idx]
        edge_state.ball_storage[start : start+len(ball)] = ball
        
    # Metrics
    if len(lengths) > 0:
        active_lengths = lengths[active_indices]
        if len(active_lengths) > 0:
            return {
                "max": int(np.max(active_lengths)),
                "mean": float(np.mean(active_lengths)),
                "p95": float(np.percentile(active_lengths, 95))
            }
    return {"max": 0, "mean": 0.0, "p95": 0.0}