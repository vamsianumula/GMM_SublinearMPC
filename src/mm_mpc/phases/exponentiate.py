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
    vertex_state: EdgeState,
    config: MPCConfig, 
    participating_mask: Optional[np.ndarray] = None
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
                
        recv_data = mpi_helpers.exchange_buffers(comm, send_bufs, dtype=np.int64)
        
        # 2. Vertex Aggregation (Using VertexState)
        # We need to map incoming 'v' to local row index.
        # Then, for each 'v', we collect the incoming balls.
        # AND we need to add the incident edges of 'v' that are in H_s.
        # Wait, the exponentiation step says:
        # "For each ball-id e that touches v, we want to add all incident sparse edges to e’s neighbor list."
        # Incident sparse edges are those in `vertex_state.adj_storage` that are ALSO participating?
        # Yes. But `adj_storage` stores local edge indices.
        # We need to check if those edges are participating.
        
        v_inbox = defaultdict(list)
        v_subscribers = defaultdict(list)
        
        for r_buf in recv_data:
            cursor = 0
            n = len(r_buf)
            while cursor < n:
                tv, seid, length = r_buf[cursor], r_buf[cursor+1], r_buf[cursor+2]
                cursor += 3
                # We own tv (guaranteed by routing), so we process it.
                # Even if we don't have local edges incident to it.
                v_inbox[tv].append(r_buf[cursor : cursor+length])
                v_subscribers[tv].append(seid)
                cursor += length
                
        # 3. Reply to Edges
        reply_bufs = [[] for _ in range(size)]
        
        for v, balls in v_inbox.items():
            incident_eids = []
            
            # Gather incident edges from CSR (if we have any)
            if v in vertex_state.vertex_id_to_row:
                row_idx = vertex_state.vertex_id_to_row[v]
                start = vertex_state.adj_offsets[row_idx]
                end = vertex_state.adj_offsets[row_idx+1]
                local_incident_indices = vertex_state.adj_storage[start:end]
                
                for local_idx in local_incident_indices:
                    # Check if this edge is participating in this phase
                    is_participating = False
                    if participating_mask is not None:
                        if participating_mask[local_idx]:
                            is_participating = True
                    else:
                        if edge_state.active_mask[local_idx] and not edge_state.stalled[local_idx]:
                            is_participating = True
                            
                    if is_participating:
                        incident_eids.append(edge_state.edge_ids[local_idx])
            
            incident_eids_arr = np.array(incident_eids, dtype=np.int64)
            
            # Merge incoming balls + incident edges
            if not balls:
                super_b = incident_eids_arr
            else:
                # Optimization: concat then unique
                # We unite all incoming balls AND the incident edges
                combined = balls + [incident_eids_arr]
                super_b = np.unique(np.concatenate(combined))
            
            sblen = len(super_b)
            for eid in v_subscribers[v]:
                dest = hashing.get_edge_owner_from_id(eid, size)
                reply_bufs[dest].extend([eid, sblen])
                reply_bufs[dest].extend(super_b)
                
        recv_replies = mpi_helpers.exchange_buffers(comm, reply_bufs, dtype=np.int64)
        
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
        
    # Instrumentation for Sublinearity Verification
    if len(lengths) > 0:
        max_ball = np.max(lengths)
        print(f"[Metrics] Rank {rank} MaxBallSize: {max_ball}")
        
    np.cumsum(lengths, out=edge_state.ball_offsets[1:])
    
    total = edge_state.ball_offsets[-1]
    edge_state.ball_storage = np.zeros(total, dtype=np.int64)
    for idx, ball in current_balls.items():
        start = edge_state.ball_offsets[idx]
        edge_state.ball_storage[start : start+len(ball)] = ball