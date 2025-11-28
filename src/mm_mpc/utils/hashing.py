"""
src/mm_mpc/utils/hashing.py
Deterministic hashing utilities.
"""
import struct
import hashlib

_GLOBAL_SEED = 0

def init_seed(seed: int):
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

def hash64(u: int, v: int = 0, phase: int = 0, iteration: int = 0, salt: str = "") -> int:
    """
    Returns a SIGNED 64-bit integer (-2^63 to 2^63-1).
    Compatible with numpy.int64 and MPI.INT64_T.
    """
    low = u if u < v else v
    high = v if u < v else u
    
    # 'q' = signed long long (8 bytes)
    # 'Q' = unsigned long long (8 bytes)
    # We use 'q' for u,v to handle negative inputs safely
    data = struct.pack("QqqQQ", _GLOBAL_SEED, low, high, phase, iteration)
    
    if salt:
        data += salt.encode('ascii')
        
    h = hashlib.sha1(data).digest()
    # Unpack as signed 'q' to fit in standard numpy int64
    return struct.unpack("q", h[:8])[0]

def get_vertex_owner(v: int, p_size: int) -> int:
    h = hash64(v, 0, 0, 0, "vertex_owner")
    return abs(h) % p_size

def get_edge_id(u: int, v: int) -> int:
    return hash64(u, v, 0, 0, "eid")

def get_edge_owner_from_id(eid: int, p_size: int) -> int:
    """
    Determines owner based purely on Global ID.
    Used during Sparsification/Exponentiation replies.
    """
    h = hash64(eid, 0, 0, 0, "edge_owner")
    return abs(h) % p_size

def get_edge_owner(u: int, v: int, p_size: int) -> int:
    """
    Used during Graph IO loading.
    MUST match get_edge_owner_from_id(get_edge_id(u,v)).
    """
    eid = get_edge_id(u, v)
    return get_edge_owner_from_id(eid, p_size)