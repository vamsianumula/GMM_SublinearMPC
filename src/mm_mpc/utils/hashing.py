"""
src/mm_mpc/utils/hashing.py
Deterministic hashing utilities for MPC.
Enforces Symmetric Edge IDs and consistent Ownership logic.
"""

import struct
import hashlib

_GLOBAL_SEED = 0

def init_seed(seed: int):
    """Initialize the global seed for the run."""
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

def hash64(u: int, v: int = 0, phase: int = 0, iteration: int = 0, salt: str = "") -> int:
    """
    Computes a deterministic 64-bit hash (Signed INT64).
    """
    # 1. Enforce Symmetry
    low = u if u < v else v
    high = v if u < v else u
    
    # 2. Pack data "QqqQQ" -> Signed q for u, v
    data = struct.pack("QqqQQ", _GLOBAL_SEED, low, high, phase, iteration)
    
    if salt:
        data += salt.encode('ascii')
        
    h = hashlib.sha1(data).digest()
    return struct.unpack("q", h[:8])[0]

def get_vertex_owner(v: int, p_size: int) -> int:
    h = hash64(v, 0, 0, 0, "vertex_owner")
    return abs(h) % p_size

def get_edge_id(u: int, v: int) -> int:
    """Generate the canonical Global ID (Signed)."""
    return hash64(u, v, 0, 0, "eid")

def get_edge_owner_from_id(eid: int, p_size: int) -> int:
    """
    Canonical logic for edge ownership.
    Depends ONLY on the Global ID.
    """
    h = hash64(eid, 0, 0, 0, "edge_owner")
    return abs(h) % p_size

def get_edge_owner(u: int, v: int, p_size: int) -> int:
    """
    Helper for graph loading. 
    Computes ID first, then Owner, ensuring consistency.
    """
    eid = get_edge_id(u, v)
    return get_edge_owner_from_id(eid, p_size)