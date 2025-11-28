"""
src/mm_mpc/phases/stall.py
Phase 2: Stalling Logic.
"""

import numpy as np
from ..state_layout import EdgeState
from ..config import MPCConfig

def apply_stalling(
    edge_state: EdgeState, 
    phase: int, 
    config: MPCConfig
) -> None:
    """
    Updates the 'stalled' flag. 
    If deg_in_sparse > T, edge is stalled.
    
    Enforces Monotonicity: Once stalled in a phase, stays stalled.
    """
    if config.R_rounds > 0:
        exponent = 1.0 / config.R_rounds
        # T = S^(1/R)
        threshold = int(config.S_edges ** exponent)
        threshold = max(threshold, 2)
    else:
        threshold = config.S_edges
        
    over_threshold = edge_state.deg_in_sparse > threshold
    
    # Only active edges are candidates for stalling
    new_stalls = over_threshold & edge_state.active_mask
    
    # Monotonic update: Union with existing stalls
    # This prevents an edge from "unstalling" if degree variance occurs
    edge_state.stalled[:] |= new_stalls