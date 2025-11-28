"""
src/mm_mpc/utils/indexing.py
Utilities for mapping Global IDs to Local Indices and managing CSR structures.
"""

import numpy as np
from typing import Dict, Tuple

def build_id_to_index_map(ids: np.ndarray) -> Dict[int, int]:
    """
    Builds a dictionary mapping Global ID -> Local Array Index.
    
    Args:
        ids: Array of global IDs (e.g., edge_ids or vertex_ids).
        
    Returns:
        Dict[int, int]: Map where d[global_id] = local_index.
    """
    # Using a dict is acceptable here because:
    # 1. It provides O(1) lookup which is mandatory for message processing.
    # 2. It is rebuilt only during compaction phases, not inside inner loops.
    return {gid: i for i, gid in enumerate(ids)}

def local_indices_from_global(global_ids: np.ndarray, lookup_map: Dict[int, int]) -> np.ndarray:
    """
    Vectorized-style lookup for a batch of global IDs.
    
    Args:
        global_ids: Array of global IDs to look up.
        lookup_map: The id_to_index dictionary.
        
    Returns:
        np.ndarray: Local indices. 
    """
    # Note: List comprehension is faster than np.vectorize for dict lookups in Python
    # We cast back to numpy for consistent type handling
    try:
        return np.array([lookup_map[gid] for gid in global_ids], dtype=np.int32)
    except KeyError as e:
        raise ValueError(f"CRITICAL: Global ID {e} not found in local lookup map. "
                         f"Topology mismatch or ownership violation.")

def build_csr_from_adj_list(num_nodes: int, adjacency_list: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper to convert a list-of-lists adjacency to CSR arrays.
    Used during initialization or compaction.
    
    Args:
        num_nodes: Number of nodes (rows).
        adjacency_list: List of length num_nodes, where each element is a list of neighbors.
        
    Returns:
        (offsets, storage): CSR arrays.
    """
    offsets = np.zeros(num_nodes + 1, dtype=np.int64)
    
    # Calculate lengths
    lengths = np.array([len(adj) for adj in adjacency_list], dtype=np.int32)
    
    # Prefix sum for offsets
    np.cumsum(lengths, out=offsets[1:])
    
    # Flatten storage
    if len(adjacency_list) > 0:
        # Check if list is empty or list of empty lists
        flat = [item for sublist in adjacency_list for item in sublist]
        storage = np.array(flat, dtype=np.int64) # Global IDs are int64
    else:
        storage = np.array([], dtype=np.int64)
        
    return offsets, storage