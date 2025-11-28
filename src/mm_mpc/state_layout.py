import numpy as np
from dataclasses import dataclass
from typing import Dict

@dataclass
class EdgeState:
    edges_local: np.ndarray    # (m, 2) int64
    edge_ids: np.ndarray       # (m,) int64
    active_mask: np.ndarray    # (m,) bool
    deg_in_sparse: np.ndarray  # (m,) int32
    stalled: np.ndarray        # (m,) bool
    priority: np.ndarray       # (m,) uint64
    ball_offsets: np.ndarray   # (m+1,) int64
    ball_storage: np.ndarray   # flat int64
    id_to_index: Dict[int, int]
    matched_edge: np.ndarray   # (m,) bool

def init_edge_state(edges_np: np.ndarray, global_ids: np.ndarray) -> EdgeState:
    m = len(edges_np)
    id_map = {gid: i for i, gid in enumerate(global_ids)}
    
    return EdgeState(
        edges_local=edges_np.astype(np.int64),
        edge_ids=global_ids.astype(np.int64),
        active_mask=np.ones(m, dtype=bool),
        deg_in_sparse=np.zeros(m, dtype=np.int32),
        stalled=np.zeros(m, dtype=bool),
        priority=np.zeros(m, dtype=np.uint64),
        ball_offsets=np.zeros(m + 1, dtype=np.int64),
        ball_storage=np.array([], dtype=np.int64),
        id_to_index=id_map,
        matched_edge=np.zeros(m, dtype=bool)
    )