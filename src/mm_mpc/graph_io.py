import sys
import numpy as np
from mpi4py import MPI
from typing import Tuple, List
from .utils import hashing

def load_and_distribute_graph(comm: MPI.Comm, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    send_counts = None
    send_data = None
    displs = None
    
    if rank == 0:
        print(f"[IO] Loading {filepath}...")
        buckets: List[List[int]] = [[] for _ in range(size)]
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip(): continue
                    parts = line.split()
                    if len(parts) < 2: continue
                    u, v = int(parts[0]), int(parts[1])
                    owner = hashing.get_edge_owner(u, v, size)
                    buckets[owner].extend([u, v])
            
            counts_list = [len(b) for b in buckets]
            send_counts = np.array(counts_list, dtype=np.int32)
            displs = np.concatenate(([0], np.cumsum(send_counts)[:-1])).astype(np.int32)
            flat_buckets = [np.array(b, dtype=np.int64) for b in buckets]
            send_data = np.concatenate(flat_buckets) if flat_buckets else np.array([], dtype=np.int64)
        except Exception as e:
            print(f"[IO] Error: {e}")
            comm.Abort(1)

    my_count_buf = np.zeros(1, dtype=np.int32)
    comm.Scatter(send_counts, my_count_buf, root=0)
    recv_buffer = np.empty(my_count_buf[0], dtype=np.int64)
    comm.Scatterv([send_data, send_counts, displs, MPI.INT64_T], recv_buffer, root=0)
    
    m_local = my_count_buf[0] // 2
    local_edges = recv_buffer.reshape((m_local, 2))
    
    local_ids = np.empty(m_local, dtype=np.int64)
    for i in range(m_local):
        local_ids[i] = hashing.get_edge_id(local_edges[i,0], local_edges[i,1])
        
    return local_edges, local_ids