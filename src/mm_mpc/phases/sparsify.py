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

def compute_deg_in_sparse(comm: MPI.Comm, edge_state: EdgeState, participating_mask: np.ndarray, p_size: int):
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
    
    # 2. Vertex Count
    v_counts = defaultdict(int)
    v_requests = defaultdict(list)
    for r_data in recv_lists:
        n = len(r_data)
        for k in range(0, n, 2):
            v, eid = r_data[k], r_data[k+1]
            v_counts[v] += 1
            v_requests[v].append(eid)
            
    # 3. Reply to Edge
    reply_bufs = [[] for _ in range(p_size)]
    for v, count in v_counts.items():
        for eid in v_requests[v]:
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
        
    # Metrics
    active_indices = np.where(participating_mask)[0]
    if len(active_indices) > 0:
        degs = edge_state.deg_in_sparse[active_indices]
        return {
            "min": float(np.min(degs)),
            "max": float(np.max(degs)),
            "mean": float(np.mean(degs)),
            "p95": float(np.percentile(degs, 95))
        }
    return {"min": 0.0, "max": 0.0, "mean": 0.0, "p95": 0.0}