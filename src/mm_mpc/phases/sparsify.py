"""
src/mm_mpc/phases/sparsify.py
Phase 1: Implicit Line Graph Sparsification.
"""

import numpy as np
from mpi4py import MPI
from collections import defaultdict
from ..state_layout import EdgeState
from ..utils import hashing, mpi_helpers

def compute_phase_participation(
    edge_state: EdgeState, 
    phase: int, 
    iteration: int,
    p_val: float
) -> np.ndarray:
    """
    Determines which active edges participate in this iteration.
    """
    m = len(edge_state.edges_local)
    participating = np.zeros(m, dtype=bool)
    
    # p_val check mapped to signed int64 range
    # abs(hash) / 2^63 <= p_val
    limit = int(p_val * 9223372036854775807) 
    
    eids = edge_state.edge_ids
    active = edge_state.active_mask
    stalled = edge_state.stalled
    
    for i in range(m):
        if active[i] and not stalled[i]:
            h = hashing.hash64(eids[i], 0, phase, iteration, "sample")
            if abs(h) <= limit:
                participating[i] = True
                
    return participating

def compute_deg_in_sparse(
    comm: MPI.Comm,
    edge_state: EdgeState,
    participating_mask: np.ndarray,
    p_size: int
) -> None:
    """
    Computes degree of each edge in the sparse line graph.
    deg_L(u,v) = (d_u - 1) + (d_v - 1)
    """
    # --- Step 1: Edge -> Vertex (Notification) ---
    local_indices = np.where(participating_mask)[0]
    m_participating = len(local_indices)
    
    active_edges = edge_state.edges_local[local_indices]
    active_eids = edge_state.edge_ids[local_indices]
    
    send_bufs = [[] for _ in range(p_size)]
    
    for i in range(m_participating):
        u, v = active_edges[i]
        eid = active_eids[i]
        
        owner_u = hashing.get_vertex_owner(u, p_size)
        owner_v = hashing.get_vertex_owner(v, p_size)
        
        # Notify both endpoints
        send_bufs[owner_u].extend([u, eid])
        send_bufs[owner_v].extend([v, eid])
        
    recv_lists = mpi_helpers.exchange_buffers(comm, send_bufs, dtype=np.int64)
    
    # --- Step 2: Vertex Counting ---
    v_counts = defaultdict(int)
    v_requests = defaultdict(list)
    
    for r_data in recv_lists:
        n_items = len(r_data)
        if n_items == 0: continue
        
        # Pairs of (v, eid)
        for k in range(0, n_items, 2):
            v = r_data[k]
            eid = r_data[k+1]
            
            v_counts[v] += 1
            v_requests[v].append(eid)
            
    # --- Step 3: Vertex -> Edge (Reply Counts) ---
    reply_bufs = [[] for _ in range(p_size)]
    
    for v, count in v_counts.items():
        requesting_eids = v_requests[v]
        
        for eid in requesting_eids:
            # CORRECTED ROUTING: Use EID to find owner
            dest_rank = hashing.get_edge_owner_from_id(eid, p_size)
            
            # Payload: [target_eid, count_at_v]
            reply_bufs[dest_rank].extend([eid, count])
            
    recv_replies = mpi_helpers.exchange_buffers(comm, reply_bufs, dtype=np.int64)
    
    # --- Step 4: Edge Update ---
    # Reset degrees for current participating edges
    # We map back using id_to_index
    
    # Important: We must zero out degrees for participating edges first
    # Otherwise, if we don't get a reply (shouldn't happen), we might keep old data
    # or if we are re-calculating.
    # Actually, simpler to zero out only the ones we update.
    
    lookup = edge_state.id_to_index
    accumulators = defaultdict(int)
    
    for r_data in recv_replies:
        n_items = len(r_data)
        for k in range(0, n_items, 2):
            eid = r_data[k]
            count = r_data[k+1]
            
            if eid in lookup:
                local_idx = lookup[eid]
                accumulators[local_idx] += count
                
    # Apply updates
    # Only update edges that actually participated
    edge_state.deg_in_sparse[local_indices] = 0 # Clear old
    
    for idx, total_degree_sum in accumulators.items():
        # deg_L(e) = d_u + d_v - 2
        final_deg = total_degree_sum - 2
        edge_state.deg_in_sparse[idx] = max(0, final_deg)