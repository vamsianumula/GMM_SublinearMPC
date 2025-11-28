import numpy as np
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

def run_greedy_mis(edge_state: EdgeState, phase: int) -> np.ndarray:
    m = len(edge_state.edges_local)
    chosen = np.zeros(m, dtype=bool)
    candidates = np.where(edge_state.active_mask & ~edge_state.stalled)[0]
    
    for idx in candidates:
        my_eid = edge_state.edge_ids[idx]
        my_prio = to_uint64(hashing.hash64(my_eid, 0, phase, 0, "priority"))
        
        is_local_max = True
        start = edge_state.ball_offsets[idx]
        end = edge_state.ball_offsets[idx+1]
        
        if start == end:
            chosen[idx] = True
            continue
            
        for n_eid in edge_state.ball_storage[start:end]:
            if n_eid == my_eid: continue
            
            n_prio = to_uint64(hashing.hash64(n_eid, 0, phase, 0, "priority"))
            
            if n_prio > my_prio:
                is_local_max = False; break
            elif n_prio == my_prio and n_eid > my_eid:
                is_local_max = False; break
        
        if is_local_max: chosen[idx] = True
            
    return chosen