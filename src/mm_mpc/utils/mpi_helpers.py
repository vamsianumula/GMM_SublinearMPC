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
    
    Architecture:
    1. Metadata Exchange: Everyone tells receivers how big the FINAL total payload is.
    2. Allocation: Receivers allocate the final destination arrays (safe due to Stalling logic).
    3. Streaming Loop: Senders transmit data in MAX_CHUNK_BYTES slices until done.
    
    Args:
        comm: MPI Communicator
        send_buffers: List of lists (data to send to each rank)
        dtype: Numpy data type
        tracker: Optional MetricsTracker for observability
        
    Returns:
        List[np.ndarray]: Data received from each rank.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # --- 1. Preparation & Flattening ---
    # We convert the list-of-lists to a list-of-arrays to avoid Python object overhead
    # Note: We do NOT concatenate everything yet, as that might double memory usage.
    # We inspect sizes first.
    
    local_send_counts = np.array([len(b) for b in send_buffers], dtype=np.int32)
    total_send_elements = np.sum(local_send_counts)
    
    # Calculate item size to determine chunking limits
    item_size = np.dtype(dtype).itemsize
    max_elements_per_chunk = MAX_CHUNK_BYTES // item_size
    
    # Flatten source data?
    # Trade-off: 
    # A) Flattening creates a copy (Double RAM usage). 
    # B) Sending from list-of-arrays requires complex offset logic.
    # Given 'Strongly Sublinear' constraints, we usually assume we have S_edges working RAM.
    # We flatten for efficiency, assuming send_buffers fits in RAM.
    # If send_buffers is HUGE, this line itself is the OOM risk.
    # However, 'stalling' should keep send_buffers <= O(S).
    if total_send_elements > 0:
        flat_send_data = np.concatenate([np.array(b, dtype=dtype) for b in send_buffers])
    else:
        flat_send_data = np.array([], dtype=dtype)

    # INSTRUMENTATION
    if tracker:
        tracker.record_comm(flat_send_data.nbytes)

    # --- 2. Metadata Exchange (The "Promise") ---
    # Tell receivers how much memory they MUST allocate to hold the result.
    global_recv_counts = np.zeros(size, dtype=np.int32)
    comm.Alltoall(local_send_counts, global_recv_counts)
    
    # Fail-Fast: If a receiver is told to allocate > MEM_LIMIT, crash now.
    # (Optional: check against psutil.virtual_memory() here)
    
    # Allocate Final Output Buffers
    recv_buffers = [np.empty(count, dtype=dtype) for count in global_recv_counts]
    
    # Trackers for the streaming loop
    # send_displs: where does rank i's data start in my flat_send_data?
    send_displs = np.concatenate(([0], np.cumsum(local_send_counts)[:-1])).astype(np.int32)
    
    # cursors: how many elements have I sent/received so far?
    send_cursors = np.zeros(size, dtype=np.int32)
    recv_cursors = np.zeros(size, dtype=np.int32)

    # --- 3. The Streaming Loop ---
    mpi_type = MPI.INT64_T if dtype == np.int64 else MPI.INT32_T
    if dtype == np.uint64: mpi_type = MPI.UINT64_T
    if dtype == np.float64: mpi_type = MPI.DOUBLE

    while True:
        # A. Determine Payload for this Micro-Step
        # Each rank calculates how much it can send to every other rank 
        # without exceeding CHUNK limit per message.
        
        current_send_counts = np.zeros(size, dtype=np.int32)
        active_sender = 0
        
        # Build the chunk to send
        send_chunks = []
        
        for dest in range(size):
            remaining = local_send_counts[dest] - send_cursors[dest]
            if remaining > 0:
                # We send the minimum of "what's left" and "chunk limit"
                count = min(remaining, max_elements_per_chunk)
                current_send_counts[dest] = count
                
                # Extract slice (View, no copy if possible)
                start = send_displs[dest] + send_cursors[dest]
                send_chunks.append(flat_send_data[start : start+count])
                
                # Advance cursor logic locally
                send_cursors[dest] += count
                active_sender = 1
            else:
                send_chunks.append(np.array([], dtype=dtype))
        
        # B. Global Termination Check
        # If NOBODY has anything left to send, we are done.
        any_active = comm.allreduce(active_sender, op=MPI.MAX)
        if not any_active:
            break
            
        # C. Prepare MPI Payloads
        if len(send_chunks) > 0:
            flat_chunk_send = np.concatenate(send_chunks)
        else:
            flat_chunk_send = np.array([], dtype=dtype)
            
        # Displacements for this specific Alltoallv call
        chunk_send_displs = np.concatenate(([0], np.cumsum(current_send_counts)[:-1])).astype(np.int32)
        
        # D. Exchange Chunk Metadata
        # Tell receivers how big *this specific chunk* is
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
        
        # G. Unpack / Reconstruction
        # Copy the received chunk data into the correct positions in the final buffers
        cursor = 0
        for src in range(size):
            count = current_recv_counts[src]
            if count > 0:
                data = flat_chunk_recv[cursor : cursor+count]
                
                # Write into pre-allocated final buffer
                start_idx = recv_cursors[src]
                recv_buffers[src][start_idx : start_idx+count] = data
                
                recv_cursors[src] += count
                cursor += count
                
    return recv_buffers