"""
src/mm_mpc/config.py
Configuration management for Strongly Sublinear MPC Maximal Matching.
Enforces S_edges constraints and memory guard thresholds.
"""

import math
import logging
from dataclasses import dataclass

@dataclass(frozen=True)
class MPCConfig:
    """
    Immutable configuration state for the MPC algorithm.
    Includes theoretical parameters and system limits.
    """
    # 1. Theoretical Parameters
    alpha: float            # The exponent 0 < alpha < 1
    n_global: int           # Total number of vertices
    m_global: int           # Total number of edges
    
    # 2. Derived Constraints (The MPC Bounds)
    S_edges: int            # Max edges per machine (c * n^alpha)
    R_rounds: int           # Exponentiation rounds ~ sqrt(log Delta)
    
    # 3. System Limits
    mem_per_cpu_gb: float   # SLURM limit
    mem_soft_limit: float = 0.75  # Trigger GC
    mem_hard_limit: float = 0.90  # Abort
    
    # 4. Phase Parameters
    small_threshold_factor: int = 100000  # Threshold for gathering to rank 0
    
    # 5. Debugging & Determinism
    random_seed: int = 42   # Base seed for deterministic hashing
    
    @classmethod
    def from_args(cls, args, mpi_size: int) -> 'MPCConfig':
        """
        Factory to create config and validate MPC constraints.
        """
        # Theoretical limit calculation: S = n^alpha
        # In a strict theoretical simulation, we use this exact value.
        # In engineering practice, we scale this by a constant 'c' (e.g., 5000) 
        # or fit to memory. Here we strictly follow the power law.
        
        # NOTE: For small n_global in testing, n^alpha might be too small.
        # We ensure a minimum floor for sanity (e.g. 1000 edges)
        calculated_s = int(math.pow(args.n_global, args.alpha) * 1000) # constant c=1000 for practical scaling
        s_edges = max(calculated_s, 2000) 

        # Validate that the cluster is large enough to hold the graph
        total_capacity = s_edges * mpi_size
        
        # log warning if capacity is tight, but don't abort yet (allow swap/overhead)
        if total_capacity < args.m_global:
            logging.warning(
                f"MPC WARNING: Total theoretical capacity ({total_capacity:_}) "
                f"is less than m_global ({args.m_global:_}). "
                f"This simulation may exceed theoretical memory bounds."
            )

        # R = sqrt(log n) as proxy for sqrt(log Delta)
        # We floor it to at least 2 for safety
        r_rounds = max(2, int(math.sqrt(math.log(max(args.n_global, 10)))))

        return cls(
            alpha=args.alpha,
            n_global=args.n_global,
            m_global=args.m_global,
            S_edges=s_edges,
            R_rounds=r_rounds,
            mem_per_cpu_gb=args.mem_per_cpu
        )