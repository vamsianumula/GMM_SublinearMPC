import math
import logging
from dataclasses import dataclass

@dataclass(frozen=True)
class MPCConfig:
    alpha: float
    n_global: int
    m_global: int
    S_edges: int
    R_rounds: int
    mem_per_cpu_gb: float
    mem_soft_limit: float = 0.75
    mem_hard_limit: float = 0.90
    small_threshold_factor: int = 100000
    random_seed: int = 42
    
    @classmethod
    def from_args(cls, args, mpi_size: int) -> 'MPCConfig':
        # Theoretical S = n^alpha
        # Practical floor of 2000 prevents issues with tiny toy graphs
        if hasattr(args, 'S_edges') and args.S_edges is not None:
            s_edges = args.S_edges
        else:
            calculated_s = int(math.pow(args.n_global, args.alpha) * 1000)
            s_edges = max(calculated_s, 2000)

        # R = sqrt(log n)
        if hasattr(args, 'R_rounds') and args.R_rounds is not None:
            r_rounds = args.R_rounds
        else:
            r_rounds = max(2, int(math.sqrt(math.log(max(args.n_global, 10)))))

        return cls(
            alpha=args.alpha,
            n_global=args.n_global,
            m_global=args.m_global,
            S_edges=s_edges,
            R_rounds=r_rounds,
            mem_per_cpu_gb=args.mem_per_cpu
        )