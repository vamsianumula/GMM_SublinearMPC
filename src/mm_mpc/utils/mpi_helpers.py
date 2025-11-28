"""
src/mm_mpc/utils/mpi_helpers.py
MPI Communication patterns optimized for MPC.
Implements 'Buffer-and-Flush' to batch small messages into efficient Alltoallv calls.
"""

import numpy as np
from mpi4py import MPI
from typing import List, Any, Tuple

# Default chunk size (in bytes) to prevent integer overflows in MPI counts
# 2GB is a safe upper bound for legacy MPI implementations using 32-bit counts
MAX_CHUNK_BYTES = 1.5 * 1024**3 

def exchange_buffers(comm: MPI.Comm, send_buffers: List[List[Any]], dtype=np.int64) -> List[np.ndarray]:
    """
    Performs a buffered Alltoallv exchange.
    
    1. Flattens list-of-lists into numpy arrays.
    2. Exchanges counts (Alltoall).
    3. Exchanges data (Alltoallv).
    
    Args:
        comm: MPI communicator.
        send_buffers: List of length p, where send_buffers[dest_rank] is a list of items.
        dtype: Numpy type of the data being sent (default int64).
        
    Returns:
        List[np.ndarray]: recv_buffers, where index i is data received from rank i.
    """
    size = comm.Get_size()
    
    # 1. Prepare Send Data (Flattening)
    # We convert Python lists to Numpy arrays for efficient transmission
    np_send_buffers = [np.array(b, dtype=dtype) for b in send_buffers]
    
    # Calculate counts (number of elements)
    send_counts = np.array([len(b) for b in np_send_buffers], dtype=np.int32)
    
    # Concatenate into one big send buffer
    if sum(send_counts) > 0:
        flat_send_data = np.concatenate(np_send_buffers)
    else:
        flat_send_data = np.array([], dtype=dtype)
        
    # Check for safety against 32-bit integer overflow in MPI counts
    if flat_send_data.nbytes > MAX_CHUNK_BYTES:
        # In a full production system, we would implement chunking loops here.
        # For this reference implementation, we fail-fast.
        raise OverflowError(f"MPI Message size {flat_send_data.nbytes} exceeds safety limit.")

    # 2. Exchange Counts
    recv_counts = np.zeros(size, dtype=np.int32)
    comm.Alltoall(send_counts, recv_counts)
    
    # 3. Prepare Receive Buffer
    total_recv = np.sum(recv_counts)
    flat_recv_data = np.empty(total_recv, dtype=dtype)
    
    # Calculate displacements
    send_displs = np.concatenate(([0], np.cumsum(send_counts)[:-1])).astype(np.int32)
    recv_displs = np.concatenate(([0], np.cumsum(recv_counts)[:-1])).astype(np.int32)
    
    # 4. Perform the Exchange
    mpi_type = MPI.INT64_T if dtype == np.int64 else MPI.INT32_T # Basic type mapping
    if dtype == np.float64: mpi_type = MPI.DOUBLE
    if dtype == np.uint64: mpi_type = MPI.UINT64_T
    
    comm.Alltoallv(
        [flat_send_data, send_counts, send_displs, mpi_type],
        [flat_recv_data, recv_counts, recv_displs, mpi_type]
    )
    
    # 5. Unpack per rank (Optional, depends on consumer needs)
    # Returning list of arrays allows the caller to know source rank
    recv_buffers = []
    curr = 0
    for count in recv_counts:
        recv_buffers.append(flat_recv_data[curr : curr+count])
        curr += count
        
    return recv_buffers

def distribute_edge_data_to_vertices(
    comm: MPI.Comm,
    local_edges: np.ndarray,
    data_to_send: np.ndarray,
    vertex_owner_func
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generic pattern for Edge -> Vertex communication.
    Sends data associated with an edge to the owners of its endpoints (u and v).
    
    Args:
        local_edges: (m, 2) array of (u, v).
        data_to_send: (m, ) array of data (e.g., participation flags, global EIDs).
        vertex_owner_func: Function(v_id, p_size) -> rank.
    
    Returns:
        (recv_u_data, recv_v_data): Data received at vertex owners.
        Note: The protocol usually requires sending (u, data) or (edge_id, data).
        Here we implement a generic 'Send (Target_V, Payload)' pattern.
    """
    size = comm.Get_size()
    
    # Buffers: [rank] -> [v_id, payload, v_id, payload...]
    # We interleave target_vertex and payload to keep streams aligned
    send_bufs = [[] for _ in range(size)]
    
    m = len(local_edges)
    p_size = size
    
    for i in range(m):
        u, v = local_edges[i]
        payload = data_to_send[i]
        
        # To owner of U
        rank_u = vertex_owner_func(u, p_size)
        send_bufs[rank_u].extend([u, payload])
        
        # To owner of V
        rank_v = vertex_owner_func(v, p_size)
        send_bufs[rank_v].extend([v, payload])
        
    # Exchange
    recv_raw_list = exchange_buffers(comm, send_bufs, dtype=np.int64)
    
    # Caller is responsible for parsing the interleaved (v, payload) stream
    return recv_raw_list