import struct
import hashlib

_GLOBAL_SEED = 0

def init_seed(seed: int):
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

def hash64(u: int, v: int = 0, phase: int = 0, iteration: int = 0, salt: str = "") -> int:
    """
    Returns a SIGNED 64-bit integer (-2^63 to 2^63-1).
    Consistent with numpy.int64 and MPI.INT64_T.
    """
    low = u if u < v else v
    high = v if u < v else u
    
    # Q = unsigned 8 bytes, q = signed 8 bytes
    # u, v can be IDs which might be large/negative, so use 'q'
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
    """Canonical Global ID (Signed)."""
    return hash64(u, v, 0, 0, "eid")

def get_edge_owner_from_id(eid: int, p_size: int) -> int:
    """Canonical Owner logic based ONLY on Global ID."""
    h = hash64(eid, 0, 0, 0, "edge_owner")
    return abs(h) % p_size

def get_edge_owner(u: int, v: int, p_size: int) -> int:
    """Consistent helper for loading."""
    eid = get_edge_id(u, v)
    return get_edge_owner_from_id(eid, p_size)