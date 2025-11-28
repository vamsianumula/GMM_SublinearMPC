import numpy as np
from mpi4py import MPI
from typing import List, Any

# 1.5 GB limit per chunk to be safe
MAX_CHUNK_BYTES = 1.5 * 1024**3 

def exchange_buffers(comm: MPI.Comm, send_buffers: List[List[Any]], dtype=np.int64) -> List[np.ndarray]:
    size = comm.Get_size()
    np_send_buffers = [np.array(b, dtype=dtype) for b in send_buffers]
    send_counts = np.array([len(b) for b in np_send_buffers], dtype=np.int32)
    
    if sum(send_counts) > 0:
        flat_send_data = np.concatenate(np_send_buffers)
    else:
        flat_send_data = np.array([], dtype=dtype)
        
    if flat_send_data.nbytes > MAX_CHUNK_BYTES:
        raise OverflowError("MPI Message size exceeds safety limit.")

    recv_counts = np.zeros(size, dtype=np.int32)
    comm.Alltoall(send_counts, recv_counts)
    
    total_recv = np.sum(recv_counts)
    flat_recv_data = np.empty(total_recv, dtype=dtype)
    
    send_displs = np.concatenate(([0], np.cumsum(send_counts)[:-1])).astype(np.int32)
    recv_displs = np.concatenate(([0], np.cumsum(recv_counts)[:-1])).astype(np.int32)
    
    mpi_type = MPI.INT64_T if dtype == np.int64 else MPI.INT32_T
    if dtype == np.uint64: mpi_type = MPI.UINT64_T
    
    comm.Alltoallv(
        [flat_send_data, send_counts, send_displs, mpi_type],
        [flat_recv_data, recv_counts, recv_displs, mpi_type]
    )
    
    recv_buffers = []
    curr = 0
    for count in recv_counts:
        recv_buffers.append(flat_recv_data[curr : curr+count])
        curr += count
        
    return recv_buffers