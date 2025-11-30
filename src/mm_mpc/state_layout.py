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

@dataclass
class VertexState:
    vertex_ids: np.ndarray       # (n_local,) int64
    vertex_id_to_row: Dict[int, int]
    adj_offsets: np.ndarray      # (n_local+1,) int64
    adj_storage: np.ndarray      # (total_incident,) int32 (local edge indices)

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

def init_vertex_state(comm, edge_state: EdgeState) -> VertexState:
    from .utils import hashing
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 1. Identify owned vertices from local edges
    # We only know about vertices that appear in our local edges.
    # In a full MPC model, we might own vertices that have NO local edges, 
    # but for this algorithm (edge-centric), we only care about vertices incident to local edges
    # because we need to map edges to them. 
    # WAIT: The algorithm requires us to aggregate info for ALL edges incident to an owned vertex.
    # Since edges are distributed by edge_owner, an owned vertex v might appear on multiple ranks.
    # But we only store the VertexState on the rank that OWNS v.
    # AND that rank must know about ALL incident edges?
    # No, the "Vertex Processing" step receives messages from ALL ranks about incident edges.
    # So we just need to know which vertices we own.
    # But we don't have a list of "all vertices in the graph".
    # We can only discover vertices that appear in our partition?
    # NO. The standard MPC assumption is we process messages for vertices we own.
    # If we receive a message for v, and we own v, we process it.
    # But we need a local mapping for `vertex_ids` to `row`.
    # We can build this dynamically or pre-scan.
    # However, `dev_reference` says "Stored only for vertices owned on each rank".
    # And "adjacency: dict[v] = list(local edge indices)".
    # This implies we only store adjacency for local edges that touch our owned vertices.
    # Let's scan local edges, find u, v. If owner(u) == me, add to u's list.
    
    local_adj = {} # vid -> list of local edge indices
    
    m = len(edge_state.edges_local)
    for i in range(m):
        u, v = edge_state.edges_local[i]
        
        if hashing.get_vertex_owner(u, size) == rank:
            if u not in local_adj: local_adj[u] = []
            local_adj[u].append(i)
            
        if hashing.get_vertex_owner(v, size) == rank:
            if v not in local_adj: local_adj[v] = []
            local_adj[v].append(i)
            
    # Convert to CSR
    sorted_vids = sorted(local_adj.keys())
    n_local = len(sorted_vids)
    
    vertex_ids = np.array(sorted_vids, dtype=np.int64)
    vertex_id_to_row = {vid: i for i, vid in enumerate(sorted_vids)}
    
    adj_offsets = np.zeros(n_local + 1, dtype=np.int64)
    
    # Calculate offsets
    current_offset = 0
    for i, vid in enumerate(sorted_vids):
        adj_offsets[i] = current_offset
        current_offset += len(local_adj[vid])
    adj_offsets[n_local] = current_offset
    
    # Fill storage
    total_edges = adj_offsets[-1]
    adj_storage = np.empty(total_edges, dtype=np.int32)
    
    for i, vid in enumerate(sorted_vids):
        start = adj_offsets[i]
        edges = local_adj[vid]
        adj_storage[start : start + len(edges)] = edges
        
    return VertexState(
        vertex_ids=vertex_ids,
        vertex_id_to_row=vertex_id_to_row,
        adj_offsets=adj_offsets,
        adj_storage=adj_storage
    )