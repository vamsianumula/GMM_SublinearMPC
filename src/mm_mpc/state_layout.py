"""
src/mm_mpc/state_layout.py
Core data structures for EdgeState and VertexState.
Implements Compressed Sparse Row (CSR) layout for vertices to respect memory limits.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class EdgeState:
    """
    SOA (Struct of Arrays) layout for Edges stored on this rank.
    """
    # Topology
    edges_local: np.ndarray    # (m_local, 2) int64: Endpoints (u, v)
    edge_ids: np.ndarray       # (m_local,) int64: Global EIDs
    
    # Algorithm State
    active_mask: np.ndarray    # (m_local,) bool: Is edge still in graph?
    deg_in_sparse: np.ndarray  # (m_local,) int32: Estimated degree in L(G)
    stalled: np.ndarray        # (m_local,) bool: Stalled status
    priority: np.ndarray       # (m_local,) uint64: Deterministic priority
    
    # Ball Storage (CSR-like flattened representation)
    # balls[i] corresponds to edges_local[i]
    # To save memory, we might only store balls for *active* edges, 
    # but for simplicity in v1, we allocate offsets for all.
    ball_offsets: np.ndarray   # (m_local + 1,) int64
    ball_storage: np.ndarray   # (total_ball_items,) int64 (Stores Global EIDs)
    
    # Lookup
    # Maps Global_EID -> Local_Index
    # Essential for mapping incoming messages back to local arrays
    id_to_index: Dict[int, int]

@dataclass
class VertexState:
    """
    CSR layout for Vertices owned by this rank.
    """
    vertex_ids: np.ndarray     # (n_local,) int64: Vertices owned here
    
    # CSR Adjacency
    # Stores indices into EdgeState on the owner rank (if edge is local) 
    # OR Global EIDs if we are tracking incident edges from other ranks.
    # Note: In the Ghaffari-Uitto model, VertexState aggregates info.
    # We store the Global EIDs of incident edges here.
    adj_offsets: np.ndarray    # (n_local + 1,) int64
    adj_storage: np.ndarray    # (total_incident,) int64 (Global EIDs)
    
    # Lookup
    # Maps Global_VID -> Local_Row_Index
    vertex_id_to_row: Dict[int, int]
    
    # State
    matched_vertex: np.ndarray # (n_local,) bool

def init_edge_state(edges_np: np.ndarray, global_ids: np.ndarray) -> EdgeState:
    """
    Initialize EdgeState from loaded raw edges.
    """
    m = len(edges_np)
    
    # Build O(1) lookup
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
        id_to_index=id_map
    )

def init_vertex_state_csr(
    vertex_ids: np.ndarray, 
    incident_counts: np.ndarray, 
    flat_adjacencies: np.ndarray
) -> VertexState:
    """
    Constructs VertexState given pre-calculated adjacency data.
    This data usually comes from an MPI shuffle (implemented in later batches).
    """
    n = len(vertex_ids)
    
    # 1. Build Offsets from counts
    offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(incident_counts, out=offsets[1:])
    
    # 2. Build Map
    v_map = {vid: i for i, vid in enumerate(vertex_ids)}
    
    return VertexState(
        vertex_ids=vertex_ids.astype(np.int64),
        adj_offsets=offsets,
        adj_storage=flat_adjacencies.astype(np.int64),
        vertex_id_to_row=v_map,
        matched_vertex=np.zeros(n, dtype=bool)
    )