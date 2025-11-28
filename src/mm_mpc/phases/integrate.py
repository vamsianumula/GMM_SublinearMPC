import numpy as np
from mpi4py import MPI
from ..state_layout import EdgeState
from ..utils import mpi_helpers, hashing

def update_matching_and_prune(comm: MPI.Comm, edge_state: EdgeState, chosen_mask: np.ndarray, p_size: int):
    new_indices = np.where(chosen_mask)[0]
    local_matches = []
    
    # 1. Notify Vertices
    match_notifs = [[] for _ in range(p_size)]
    for idx in new_indices:
        u, v = edge_state.edges_local[idx]
        local_matches.append((u, v))
        edge_state.matched_edge[idx] = True
        match_notifs[hashing.get_vertex_owner(u, p_size)].append(u)
        match_notifs[hashing.get_vertex_owner(v, p_size)].append(v)
        
    recv_matches = mpi_helpers.exchange_buffers(comm, match_notifs, dtype=np.int64)
    my_matched_verts = set(v for buf in recv_matches for v in buf)
    
    # 2. Query
    active_indices = np.where(edge_state.active_mask & ~edge_state.matched_edge)[0]
    query_bufs = [[] for _ in range(p_size)]
    for idx in active_indices:
        u, v = edge_state.edges_local[idx]
        eid = edge_state.edge_ids[idx]
        query_bufs[hashing.get_vertex_owner(u, p_size)].extend([u, eid])
        query_bufs[hashing.get_vertex_owner(v, p_size)].extend([v, eid])
        
    recv_queries = mpi_helpers.exchange_buffers(comm, query_bufs, dtype=np.int64)
    
    # 3. Reply Kill
    reply_bufs = [[] for _ in range(p_size)]
    for r_buf in recv_queries:
        for k in range(0, len(r_buf), 2):
            if r_buf[k] in my_matched_verts:
                dest = hashing.get_edge_owner_from_id(r_buf[k+1], p_size)
                reply_bufs[dest].append(r_buf[k+1])
                
    recv_kills = mpi_helpers.exchange_buffers(comm, reply_bufs, dtype=np.int64)
    
    # 4. Apply
    lookup = edge_state.id_to_index
    for k_buf in recv_kills:
        for eid in k_buf:
            if eid in lookup: edge_state.active_mask[lookup[eid]] = False
            
    return local_matches