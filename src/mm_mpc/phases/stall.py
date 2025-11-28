import numpy as np
from ..state_layout import EdgeState
from ..config import MPCConfig

def apply_stalling(edge_state: EdgeState, phase: int, config: MPCConfig):
    if config.R_rounds > 0:
        threshold = max(int(config.S_edges ** (1.0 / config.R_rounds)), 2)
    else:
        threshold = config.S_edges
        
    over_threshold = edge_state.deg_in_sparse > threshold
    new_stalls = over_threshold & edge_state.active_mask
    edge_state.stalled[:] |= new_stalls