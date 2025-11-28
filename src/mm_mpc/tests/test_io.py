import numpy as np
from mpi4py import MPI
from mm_mpc.graph_io import load_and_distribute_graph
from mm_mpc.utils import hashing

def test_graph_io():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Ensure determinstic hashing for the test
    hashing.init_seed(12345)
    
    # Load
    local_edges, local_ids = load_and_distribute_graph(comm, "toy.txt")
    
    # Gather to verify correctness (ONLY for small tests!)
    all_edges = comm.gather(local_edges, root=0)
    all_ids = comm.gather(local_ids, root=0)
    
    if rank == 0:
        total_edges = np.vstack(all_edges)
        total_ids = np.concatenate(all_ids)
        
        print(f"Total edges received: {len(total_edges)}")
        assert len(total_edges) == 4, f"Expected 4 edges, got {len(total_edges)}"
        
        # Check IDs match expected hash
        for i in range(4):
            u, v = total_edges[i]
            expected_id = hashing.get_edge_id(u, v)
            assert expected_id == total_ids[i], f"ID Mismatch for edge {u}-{v}"
            
        print("Graph IO Test: PASSED")

if __name__ == "__main__":
    test_graph_io()