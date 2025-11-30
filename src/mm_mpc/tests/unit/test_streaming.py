"""
tests/unit/test_streaming.py
Verifies the Chunked MPI Exchange logic.
"""
import unittest
import numpy as np
from mpi4py import MPI
from mm_mpc.utils import mpi_helpers

class TestStreaming(unittest.TestCase):
    def test_chunked_exchange(self):
        """
        Forces the exchange_buffers to loop multiple times by setting 
        a tiny MAX_CHUNK_BYTES during the test.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        # 1. Monkey-Patch the constant to force chunking
        # Set limit to 80 bytes (10 int64s)
        original_limit = mpi_helpers.MAX_CHUNK_BYTES
        mpi_helpers.MAX_CHUNK_BYTES = 80 
        
        try:
            # 2. Generate Data > 80 bytes
            # Send 20 ints (160 bytes) to neighbor
            dest = (rank + 1) % size
            payload = np.arange(20, dtype=np.int64) + (rank * 100)
            
            send_buffers = [[] for _ in range(size)]
            send_buffers[dest] = payload.tolist()
            
            # 3. Exchange
            # This should trigger at least 2 loops inside exchange_buffers
            recv_buffers = mpi_helpers.exchange_buffers(comm, send_buffers)
            
            # 4. Verify
            src = (rank - 1 + size) % size
            expected_data = np.arange(20, dtype=np.int64) + (src * 100)
            
            received = recv_buffers[src]
            
            self.assertEqual(len(received), 20)
            np.testing.assert_array_equal(received, expected_data)
            
        finally:
            # Restore constant
            mpi_helpers.MAX_CHUNK_BYTES = original_limit

if __name__ == "__main__":
    # Run with: mpirun -n 2 python3 -m mm_mpc.tests.unit.test_streaming
    unittest.main()