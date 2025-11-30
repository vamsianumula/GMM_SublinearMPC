"""
src/mm_mpc/utils/mpi_helpers.py
MPI Communication patterns.
Implements 'Chunked Exchange' (Streaming) to enforce MPC bandwidth constraints
and prevent OOM on Send buffers.
"""
import numpy as np
from mpi4py import MPI
from typing import List, Any

# Chunk size constant: 256 MB
# This is small enough to fit in L3 cache/RAM easily, but large enough to saturate bandwidth.
MAX_CHUNK_BYTES = 256 * 1024 * 1024 

def exchange_buffers(
    comm: MPI.Comm, 
    send_buffers: List[List[Any]], 
    dtype=np.int64,
    tracker=None
) -> List[np.ndarray]:
    """
    Performs a buffered Alltoallv exchange using a Streaming (Chunked) Protocol.
    
    Args:
        comm: MPI Communicator
        send_buffers: List of lists (data to send to each rank)
        dtype: Numpy data type
        tracker: Optional MetricsTracker for observability
        
    Returns:
        List[np.ndarray]: Data received from each rank.
    """
    import time
    start_time = time.time()
    
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # --- 1. Preparation & Flattening ---
    local_send_counts = np.array([len(b) for b in send_buffers], dtype=np.int32)
    total_send_elements = np.sum(local_send_counts)
    
    # Calculate item size to determine chunking limits
    item_size = np.dtype(dtype).itemsize
    max_elements_per_chunk = MAX_CHUNK_BYTES // item_size
    
    if total_send_elements > 0:
        flat_send_data = np.concatenate([np.array(b, dtype=dtype) for b in send_buffers])
    else:
        flat_send_data = np.array([], dtype=dtype)

    # --- 2. Metadata Exchange (The "Promise") ---
    global_recv_counts = np.zeros(size, dtype=np.int32)
    comm.Alltoall(local_send_counts, global_recv_counts)
    
    # Allocate Final Output Buffers
    recv_buffers = [np.empty(count, dtype=dtype) for count in global_recv_counts]
    
    # Trackers for the streaming loop
    send_displs = np.concatenate(([0], np.cumsum(local_send_counts)[:-1])).astype(np.int32)
    send_cursors = np.zeros(size, dtype=np.int32)
    recv_cursors = np.zeros(size, dtype=np.int32)

    # --- 3. The Streaming Loop ---
    mpi_type = MPI.INT64_T if dtype == np.int64 else MPI.INT32_T
    if dtype == np.uint64: mpi_type = MPI.UINT64_T
    if dtype == np.float64: mpi_type = MPI.DOUBLE

    max_msg_size_seen = 0
    total_sent_bytes = 0
    total_recv_bytes = 0

    while True:
        # A. Determine Payload for this Micro-Step
        current_send_counts = np.zeros(size, dtype=np.int32)
        active_sender = 0
        send_chunks = []
        
        for dest in range(size):
            remaining = local_send_counts[dest] - send_cursors[dest]
            if remaining > 0:
                count = min(remaining, max_elements_per_chunk)
                current_send_counts[dest] = count
                
                start = send_displs[dest] + send_cursors[dest]
                send_chunks.append(flat_send_data[start : start+count])
                
                send_cursors[dest] += count
                active_sender = 1
            else:
                send_chunks.append(np.array([], dtype=dtype))
        
        # B. Global Termination Check
        any_active = comm.allreduce(active_sender, op=MPI.MAX)
        if not any_active:
            break
            
        # C. Prepare MPI Payloads
        if len(send_chunks) > 0:
            flat_chunk_send = np.concatenate(send_chunks)
        else:
            flat_chunk_send = np.array([], dtype=dtype)
            
        chunk_send_displs = np.concatenate(([0], np.cumsum(current_send_counts)[:-1])).astype(np.int32)
        
        # D. Exchange Chunk Metadata
        current_recv_counts = np.zeros(size, dtype=np.int32)
        comm.Alltoall(current_send_counts, current_recv_counts)
        
        # E. Receive Buffer for Chunk
        total_chunk_recv = np.sum(current_recv_counts)
        flat_chunk_recv = np.empty(total_chunk_recv, dtype=dtype)
        chunk_recv_displs = np.concatenate(([0], np.cumsum(current_recv_counts)[:-1])).astype(np.int32)
        
        # F. The Physical Transfer
        comm.Alltoallv(
            [flat_chunk_send, current_send_counts, chunk_send_displs, mpi_type],
            [flat_chunk_recv, current_recv_counts, chunk_recv_displs, mpi_type]
        )
        
        # Metrics Update
        sent_b = flat_chunk_send.nbytes
        recv_b = flat_chunk_recv.nbytes
        total_sent_bytes += sent_b
        total_recv_bytes += recv_b
        max_msg_size_seen = max(max_msg_size_seen, sent_b, recv_b)
        
        # G. Unpack / Reconstruction
        cursor = 0
        for src in range(size):
            count = current_recv_counts[src]
            if count > 0:
                data = flat_chunk_recv[cursor : cursor+count]
                start_idx = recv_cursors[src]
                recv_buffers[src][start_idx : start_idx+count] = data
                recv_cursors[src] += count
                cursor += count
    
    duration = time.time() - start_time
    if tracker:
        # Calculate items (elements)
        # For variable length buffers, we can sum the counts
        total_sent_items = np.sum(local_send_counts)
        total_recv_items = np.sum(global_recv_counts)
        tracker.record_comm(total_sent_bytes, total_recv_bytes, total_sent_items, total_recv_items, max_msg_size_seen, duration)
                
    return recv_buffers