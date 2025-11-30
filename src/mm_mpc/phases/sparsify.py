import numpy as np
from mpi4py import MPI
from collections import defaultdict
from ..state_layout import EdgeState
from ..utils import hashing, mpi_helpers

def compute_phase_participation(edge_state: EdgeState, phase: int, iteration: int, p_val: float) -> np.ndarray:
    m = len(edge_state.edges_local)
    participating = np.zeros(m, dtype=bool)
    
    # abs(signed_hash) compared against max_int64 * p
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

def compute_deg_in_sparse(comm: MPI.Comm, edge_state: EdgeState, vertex_state: EdgeState, participating_mask: np.ndarray, p_size: int):
    # Note: vertex_state type hint is actually VertexState but we avoid circular import or just use Any
    edge_state.deg_in_sparse[:] = 0
    local_indices = np.where(participating_mask)[0]
    
    # 1. Edge -> Vertex
    active_edges = edge_state.edges_local[local_indices]
    active_eids = edge_state.edge_ids[local_indices]
    send_bufs = [[] for _ in range(p_size)]
    
    for i in range(len(local_indices)):
        u, v = active_edges[i]
        eid = active_eids[i]
        send_bufs[hashing.get_vertex_owner(u, p_size)].extend([u, eid])
        send_bufs[hashing.get_vertex_owner(v, p_size)].extend([v, eid])
        
    recv_lists = mpi_helpers.exchange_buffers(comm, send_bufs, dtype=np.int64)
    
    # 2. Vertex Count (Using VertexState)
    # We need to map incoming 'v' to local row index to store counts?
    # Actually, for just counting, we can use a temporary array aligned with vertex_ids
    # But since we need to reply to specific EIDs, we just need to know the count for 'v'.
    # We can use vertex_state.vertex_id_to_row to validate ownership and (if needed) store state.
    # For this phase, we just need the count.
    
    # We can use a dict for the *current phase* counts, or a dense array if we trust vertex_ids.
    # Let's use a dict for now to be safe, but populated only for owned vertices.
    # Wait, the requirement is to avoid python objects for *adjacency*. 
    # Temporary dicts for message processing are okay if they don't persist or explode.
    # But let's try to use the dense array approach if possible.
    
    n_local = len(vertex_state.vertex_ids)
    # We can't easily index by 'v' directly if v is arbitrary 64-bit.
    # We MUST use vertex_id_to_row.
    
    # 2. Vertex Count
    # We use a defaultdict to count degrees for ALL owned vertices (including ghosts).
    # This is a temporary structure for the phase, so it's allowed.
    v_counts = defaultdict(int)
    v_requests = defaultdict(list)
    
    # Pass 1: Aggregate
    for r_data in recv_lists:
        n = len(r_data)
        for k in range(0, n, 2):
            v, eid = r_data[k], r_data[k+1]
            # We own v (guaranteed by routing), so we count it.
            v_counts[v] += 1
            v_requests[v].append(eid)
            
    # 3. Reply to Edge
    reply_bufs = [[] for _ in range(p_size)]
    for v, eids in v_requests.items():
        count = v_counts[v]
        for eid in eids:
            dest = hashing.get_edge_owner_from_id(eid, p_size)
            reply_bufs[dest].extend([eid, count])
            
    recv_replies = mpi_helpers.exchange_buffers(comm, reply_bufs, dtype=np.int64)
    
    # 4. Update
    lookup = edge_state.id_to_index
    accumulators = defaultdict(int)
    for r_data in recv_replies:
        n = len(r_data)
        for k in range(0, n, 2):
            eid, count = r_data[k], r_data[k+1]
            if eid in lookup:
                accumulators[lookup[eid]] += count
                
    for idx, total in accumulators.items():
        edge_state.deg_in_sparse[idx] = max(0, total - 2)