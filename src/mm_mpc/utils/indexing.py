"""
src/mm_mpc/utils/indexing.py
Utilities for mapping Global IDs to Local Indices and managing CSR structures.
"""

import numpy as np
from typing import Dict, Tuple, List

def build_id_to_index_map(ids: np.ndarray) -> Dict[int, int]:
    """
    Builds a dictionary mapping Global ID -> Local Array Index.
    """
    return {gid: i for i, gid in enumerate(ids)}

def local_indices_from_global(global_ids: np.ndarray, lookup_map: Dict[int, int]) -> np.ndarray:
    """
    Vectorized-style lookup for a batch of global IDs.
    Returns local indices corresponding to the global IDs.
    """
    try:
        return np.array([lookup_map[gid] for gid in global_ids], dtype=np.int32)
    except KeyError as e:
        raise ValueError(f"CRITICAL: Global ID {e} not found in local lookup map. "
                         f"Topology mismatch or ownership violation.")

def build_csr_from_adj_list(num_nodes: int, adjacency_list: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper to convert a list-of-lists adjacency to CSR arrays (offsets, storage).
    """
    offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    
    # Calculate lengths
    lengths = np.array([len(adj) for adj in adjacency_list], dtype=np.int32)
    
    # Prefix sum for offsets
    np.cumsum(lengths, out=offsets[1:])
    
    # Flatten storage
    if len(adjacency_list) > 0:
        flat = [item for sublist in adjacency_list for item in sublist]
        storage = np.array(flat, dtype=np.int64)
    else:
        storage = np.array([], dtype=np.int64)
        
    return offsets, storage