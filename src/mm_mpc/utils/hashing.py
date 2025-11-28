"""
src/mm_mpc/utils/hashing.py
Deterministic hashing utilities for MPC.
Enforces Symmetric Edge IDs invariant: hash(u, v) == hash(v, u).
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
    Computes a deterministic 64-bit hash.
    
    CRITICAL INVARIANT:
    hash64(u, v, ...) MUST EQUAL hash64(v, u, ...)
    
    We achieve this by sorting u and v.
    """
    # 1. Enforce Symmetry
    # We use min/max to ensure (u,v) and (v,u) produce identical bytes
    low = u if u < v else v
    high = v if u < v else u
    
    # 2. Pack data into binary struct for speed
    # Q = unsigned long long (8 bytes)
    # We pack: seed, low, high, phase, iteration
    data = struct.pack("QQQQQ", _GLOBAL_SEED, low, high, phase, iteration)
    
    if salt:
        data += salt.encode('ascii')
        
    # 3. Compute Hash (sha1 is standard and sufficient for distribution)
    h = hashlib.sha1(data).digest()
    
    # 4. Unpack first 8 bytes as unsigned 64-bit int
    return struct.unpack("Q", h[:8])[0]

def get_vertex_owner(v: int, p_size: int) -> int:
    """Map vertex to rank."""
    # Salt ensures vertex distribution isn't identical to edge distribution
    h = hash64(v, 0, 0, 0, "vertex_owner")
    return h % p_size

def get_edge_owner(u: int, v: int, p_size: int) -> int:
    """
    Map edge to rank. 
    Symmetric by definition of hash64.
    """
    eid = hash64(u, v, 0, 0, "edge_owner")
    return eid % p_size

def get_edge_id(u: int, v: int) -> int:
    """
    Generate the canonical Global ID for an edge.
    Symmetric by definition of hash64.
    """
    return hash64(u, v, 0, 0, "eid")