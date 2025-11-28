"""
src/mm_mpc/phases/integrate.py
Phase 5: Integration. Removes edges incident to matched vertices.
"""

import numpy as np
from mpi4py import MPI
from ..state_layout import EdgeState
from ..utils import mpi_helpers, hashing

def update_matching_and_prune(
    comm: MPI.Comm,
    edge_state: EdgeState,
    chosen_mask: np.ndarray,
    p_size: int
) -> list:
    """
    Returns list of newly matched (u, v) tuples.
    Updates edge_state.active_mask to remove incident edges AND chosen edges.
    """
    # 1. Identify newly matched edges
    new_indices = np.where(chosen_mask)[0]
    local_matches = []
    
    # Notify vertex owners: "Vertex V is matched"
    match_notifs = [[] for _ in range(p_size)]
    
    for idx in new_indices:
        u, v = edge_state.edges_local[idx]
        local_matches.append((u, v))
        
        # Mark locally
        edge_state.matched_edge[idx] = True
        
        # CRITICAL FIX: Chosen edges must leave the active set immediately
        edge_state.active_mask[idx] = False 
        
        owner_u = hashing.get_vertex_owner(u, p_size)
        owner_v = hashing.get_vertex_owner(v, p_size)
        
        match_notifs[owner_u].append(u)
        match_notifs[owner_v].append(v)
        
    # Exchange 1: Notify Vertices
    recv_matches = mpi_helpers.exchange_buffers(comm, match_notifs, dtype=np.int64)
    
    my_matched_verts = set()
    for buf in recv_matches:
        for v in buf:
            my_matched_verts.add(v)
            
    # 2. Filter Active Edges (The "Kill" Phase)
    # Only check edges that are STILL active (i.e., not the ones we just chose)
    active_indices = np.where(edge_state.active_mask)[0]
    
    query_bufs = [[] for _ in range(p_size)]
    for idx in active_indices:
        u, v = edge_state.edges_local[idx]
        eid = edge_state.edge_ids[idx]
        
        owner_u = hashing.get_vertex_owner(u, p_size)
        owner_v = hashing.get_vertex_owner(v, p_size)
        
        query_bufs[owner_u].extend([u, eid])
        query_bufs[owner_v].extend([v, eid])
        
    # Exchange 2: Edges -> Vertices
    recv_queries = mpi_helpers.exchange_buffers(comm, query_bufs, dtype=np.int64)
    
    # 3. Vertices Reply
    reply_bufs = [[] for _ in range(p_size)]
    for r_buf in recv_queries:
        n = len(r_buf)
        for k in range(0, n, 2):
            target_v = r_buf[k]
            source_eid = r_buf[k+1]
            
            if target_v in my_matched_verts:
                # Reply "Kill" to edge owner
                owner_e = hashing.get_edge_owner_from_id(source_eid, p_size)
                reply_bufs[owner_e].append(source_eid)
                
    # Exchange 3: Vertices -> Edges
    recv_kills = mpi_helpers.exchange_buffers(comm, reply_bufs, dtype=np.int64)
    
    # 4. Apply Deletions
    lookup = edge_state.id_to_index
    kill_count = 0
    for k_buf in recv_kills:
        for eid in k_buf:
            if eid in lookup:
                idx = lookup[eid]
                if edge_state.active_mask[idx]:
                    edge_state.active_mask[idx] = False
                    kill_count += 1
    
    return local_matches