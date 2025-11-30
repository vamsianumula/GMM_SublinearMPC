import numpy as np
from mpi4py import MPI
from mm_mpc.state_layout import init_edge_state, init_vertex_state
from mm_mpc.phases import finish
from mm_mpc.config import MPCConfig
from mm_mpc.utils import hashing

def test_distributed_finishing():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    hashing.init_seed(42)
    
    # Setup a graph that is "too large" for sequential finish but small enough to run
    # We force global_count > threshold
    
    # Let S_edges = 10. Threshold = 10 * 0.5 = 5.
    # We create a graph with 20 edges.
    
    N = 40
    M = 20
    S_edges = 10
    
    # Create a path graph 0-1-2...20
    raw_edges = []
    for i in range(M):
        raw_edges.append((i, i+1))
        
    local_edges = []
    local_ids = []
    
    for u, v in raw_edges:
        eid = hashing.get_edge_id(u, v)
        owner = hashing.get_edge_owner_from_id(eid, size)
        if owner == rank:
            local_edges.append([u, v])
            local_ids.append(eid)
            
    local_edges = np.array(local_edges, dtype=np.int64)
    local_ids = np.array(local_ids, dtype=np.int64)
    
    if len(local_edges) == 0:
        local_edges = np.empty((0, 2), dtype=np.int64)
        
    edge_state = init_edge_state(local_edges, local_ids)
    vertex_state = init_vertex_state(comm, edge_state)
    
    config = MPCConfig(alpha=0.5, n_global=N, m_global=M, 
                       S_edges=S_edges, R_rounds=1, mem_per_cpu_gb=1.0)
    
    # Run Finish
    # It should trigger distributed fallback because M=20 > Threshold=5
    
    extra_matches = finish.finish_small_components(comm, edge_state, vertex_state, config)
    
    # Gather matches
    all_extras = comm.gather(extra_matches, root=0)
    
    if rank == 0:
        total_matches = []
        for l in all_extras:
            total_matches.extend(l)
            
        print(f"Total Matches from Finish: {len(total_matches)}")
        
        # Verify validity
        matched = set()
        for u, v in total_matches:
            assert u not in matched and v not in matched
            matched.add(u)
            matched.add(v)
            
        # Since it's a path graph, we expect a decent matching size.
        # Distributed MIS on path graph works well.
        assert len(total_matches) > 0
        
        print("Distributed Finishing Test: PASSED")

if __name__ == "__main__":
    test_distributed_finishing()
