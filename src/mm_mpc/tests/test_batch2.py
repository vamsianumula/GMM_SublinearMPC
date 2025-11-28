"""
tests/test_batch2.py
Verifies Indexing and Buffered Communication.
"""

import numpy as np
from mpi4py import MPI
from mm_mpc.utils import indexing, mpi_helpers

def test_indexing():
    # 1. Test Map Builder
    ids = np.array([101, 505, 999], dtype=np.int64)
    lookup = indexing.build_id_to_index_map(ids)
    
    assert lookup[101] == 0
    assert lookup[999] == 2
    try:
        _ = lookup[404]
        assert False, "Should raise KeyError"
    except KeyError:
        pass
        
    # 2. Test Vectorized Lookup
    targets = np.array([999, 101], dtype=np.int64)
    indices = indexing.local_indices_from_global(targets, lookup)
    assert indices[0] == 2
    assert indices[1] == 0
    
    print("Indexing Test: PASSED")

def test_mpi_buffer_exchange():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Pattern: Rank i sends [i] to Rank (i+1)%size
    dest = (rank + 1) % size
    
    send_buffers = [[] for _ in range(size)]
    send_buffers[dest].append(rank)
    
    # Exchange
    recv_buffers = mpi_helpers.exchange_buffers(comm, send_buffers)
    
    # Verify: We should receive exactly one message from (rank-1)%size containing [rank-1]
    src = (rank - 1 + size) % size
    
    received_data = recv_buffers[src]
    assert len(received_data) == 1
    assert received_data[0] == src
    
    # Verify others are empty
    for r in range(size):
        if r != src:
            assert len(recv_buffers[r]) == 0
            
    if rank == 0:
        print("MPI Exchange Test: PASSED")

if __name__ == "__main__":
    test_indexing()
    test_mpi_buffer_exchange()