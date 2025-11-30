import numpy as np
from mpi4py import MPI
from mm_mpc.state_layout import init_edge_state, init_vertex_state
from mm_mpc.utils import hashing

def test_vertex_state_csr():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    hashing.init_seed(42)
    
    # 1. Setup Edges
    # We create edges such that some vertices are owned by this rank.
    # We also want to test "ghost" vertices if possible, but init_vertex_state
    # only builds state for vertices incident to local edges.
    # Wait, the "ghost" vertex issue was about processing messages for owned vertices
    # even if they are NOT in VertexState.
    # So VertexState ONLY contains vertices with local edges.
    # This test verifies that VertexState is built correctly for local edges.
    
    # Create edges (0,1), (0,2), (1,3)
    # We force ownership to ensure some land here.
    
    local_edges = []
    local_ids = []
    
    # We'll just generate a bunch of edges and check consistency
    for u, v in [(0, 1), (0, 2), (1, 3), (2, 4), (4, 5)]:
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
    
    # Verify CSR Properties
    assert len(vertex_state.vertex_ids) == len(vertex_state.vertex_id_to_row)
    assert len(vertex_state.adj_offsets) == len(vertex_state.vertex_ids) + 1
    
    # Check that for every vertex in vertex_ids, we own it
    for v in vertex_state.vertex_ids:
        assert hashing.get_vertex_owner(v, size) == rank
        
    # Check adjacency consistency
    # For each vertex v in VertexState, check that the edges in adj_storage are actually incident to v
    for i, v in enumerate(vertex_state.vertex_ids):
        start = vertex_state.adj_offsets[i]
        end = vertex_state.adj_offsets[i+1]
        edge_indices = vertex_state.adj_storage[start:end]
        
        for idx in edge_indices:
            u_edge, v_edge = edge_state.edges_local[idx]
            assert u_edge == v or v_edge == v
            
    if rank == 0:
        print("VertexState CSR Test: PASSED")

if __name__ == "__main__":
    test_vertex_state_csr()
