import numpy as np
from mpi4py import MPI
from mm_mpc.utils import hashing
from mm_mpc.state_layout import init_edge_state

def test_symmetry():
    h1 = hashing.hash64(10, 20)
    h2 = hashing.hash64(20, 10)
    assert h1 == h2, "Hashing is not symmetric!"
    print("Test Passed: Symmetry")

def test_csr_layout():
    # Simulate data
    edges = np.array([[1, 2], [3, 4]], dtype=np.int64)
    ids = np.array([100, 200], dtype=np.int64)
    
    state = init_edge_state(edges, ids)
    
    assert state.id_to_index[100] == 0
    assert state.id_to_index[200] == 1
    assert state.active_mask.all()
    print("Test Passed: CSR Layout")

if __name__ == "__main__":
    test_symmetry()
    test_csr_layout()