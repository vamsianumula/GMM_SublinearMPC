"""
src/mm_mpc/phases/local_mis.py
Phase 4: Local Maximal Independent Set on Edge Balls.
"""

import numpy as np
from typing import Optional
from ..state_layout import EdgeState
from ..utils import hashing

MASK_64 = 0xFFFFFFFFFFFFFFFF

def to_uint64(val: int) -> np.uint64:
    return np.uint64(val & MASK_64)

def assign_priorities(edge_state: EdgeState, phase: int):
    m = len(edge_state.edges_local)
    for i in range(m):
        if edge_state.active_mask[i]:
            raw = hashing.hash64(edge_state.edge_ids[i], 0, phase, 0, "priority")
            edge_state.priority[i] = to_uint64(raw)

def run_greedy_mis(
    edge_state: EdgeState, 
    phase: int,
    participating_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Selects edges for the matching based on local ball information.
    
    CRITICAL: Only edges that participated in Exponentiation (are in H_s)
    can be candidates for MIS. Edges outside H_s wait.
    """
    m = len(edge_state.edges_local)
    chosen = np.zeros(m, dtype=bool)
    
    # Candidates must be Active AND Not Stalled
    candidates_mask = edge_state.active_mask & ~edge_state.stalled
    
    # AND must have participated in the sparse graph construction
    if participating_mask is not None:
        candidates_mask &= participating_mask
        
    candidate_indices = np.where(candidates_mask)[0]
    
    for idx in candidate_indices:
        my_eid = edge_state.edge_ids[idx]
        my_prio = to_uint64(hashing.hash64(my_eid, 0, phase, 0, "priority"))
        
        is_local_max = True
        
        start = edge_state.ball_offsets[idx]
        end = edge_state.ball_offsets[idx+1]
        
        # If ball is empty (isolated in H_s), it's a local max
        if start == end:
            chosen[idx] = True
            continue
            
        ball_eids = edge_state.ball_storage[start:end]
        
        for n_eid in ball_eids:
            if n_eid == my_eid:
                continue
                
            # Check neighbor priority
            n_prio = to_uint64(hashing.hash64(n_eid, 0, phase, 0, "priority"))
            
            if n_prio > my_prio:
                is_local_max = False; break
            elif n_prio == my_prio and n_eid > my_eid:
                is_local_max = False; break
        
        if is_local_max:
            chosen[idx] = True
            
    # Metrics
    n_candidates = len(candidate_indices)
    n_chosen = np.sum(chosen)
    rate = n_chosen / n_candidates if n_candidates > 0 else 0.0
    
    return chosen, rate