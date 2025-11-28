"""
src/mm_mpc/graph_io.py
Graph loading and initial distribution.
"""

import sys
import numpy as np
from mpi4py import MPI
from typing import Tuple, List  # Fixed: Added missing imports
from .utils import hashing

def load_and_distribute_graph(comm: MPI.Comm, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads edge list from Rank 0 and distributes to owner ranks.
    
    Returns:
        local_edges: np.ndarray shape (m_local, 2)
        local_ids:   np.ndarray shape (m_local,) of Global Edge IDs
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # --- Step 1: Rank 0 Loads and Buckets Data ---
    # We initialize these to None on non-root ranks to avoid confusion
    send_counts = None
    send_data = None
    displs = None
    
    if rank == 0:
        print(f"[IO] Loading graph from {filepath}...")
        # Temporary buckets for each rank
        buckets: List[List[int]] = [[] for _ in range(size)]
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip():
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                        
                    u, v = int(parts[0]), int(parts[1])
                    
                    # Determine Owner using Symmetric Hashing
                    # This ensures the edge always lands on the same rank
                    # regardless of input order (u,v) vs (v,u)
                    owner = hashing.get_edge_owner(u, v, size)
                    
                    # Store as flat [u, v]
                    buckets[owner].extend([u, v])
            
            print("[IO] Graph loaded into RAM. Preparing MPI buffers...")
            
            # Convert buckets to flattened numpy array for Scatterv
            # 1. Calculate counts (number of INTs, which is 2 * num_edges)
            counts_list = [len(b) for b in buckets]
            send_counts = np.array(counts_list, dtype=np.int32)
            
            # 2. Calculate displacements
            displs = np.concatenate(([0], np.cumsum(send_counts)[:-1])).astype(np.int32)
            
            # 3. Flatten data
            # Note: np.concatenate is efficient for joining list of arrays
            flat_buckets = [np.array(b, dtype=np.int64) for b in buckets]
            send_data = np.concatenate(flat_buckets)
            
            print(f"[IO] Total edges: {len(send_data)//2:_}. Starting Distribution...")

        except Exception as e:
            print(f"[IO] Critical Error loading graph: {e}")
            sys.stdout.flush()
            comm.Abort(1)

    # --- Step 2: Scatter Counts (So ranks know how much to allocate) ---
    # Buffer to receive the count of integers (not edges) for this rank
    my_count_buf = np.zeros(1, dtype=np.int32)
    
    # Root sends one integer (the count) to each rank
    comm.Scatter(send_counts, my_count_buf, root=0)
    
    my_n_ints = my_count_buf[0]
    
    # --- Step 3: Allocate Receive Buffer ---
    recv_buffer = np.empty(my_n_ints, dtype=np.int64)
    
    # --- Step 4: Scatterv the Actual Edge Data ---
    # Root provides the tuple [data, counts, displs, type], others provide None
    comm.Scatterv([send_data, send_counts, displs, MPI.INT64_T], recv_buffer, root=0)
    
    # --- Step 5: Local Shaping and ID Generation ---
    m_local = my_n_ints // 2
    local_edges = recv_buffer.reshape((m_local, 2))
    
    # Compute Global IDs locally to save network bandwidth
    # This relies on hash64 being deterministic and symmetric
    local_ids = np.empty(m_local, dtype=np.int64)
    
    for i in range(m_local):
        u, v = local_edges[i]
        local_ids[i] = hashing.get_edge_id(u, v)
        
    if rank == 0:
        print(f"[IO] Distribution complete.")
        
    return local_edges, local_ids