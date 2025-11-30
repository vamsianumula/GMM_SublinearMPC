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
    
    # Adaptive Sparsification
    adaptive_sparsification: bool = True
    safety_factor: float = 1.0
    
    # Metrics
    enable_metrics: bool = True
    enable_test_mode: bool = False
    metrics_output_dir: str = "experiments/results/latest"
    
    @classmethod
    def from_args(cls, args, mpi_size: int) -> 'MPCConfig':
        # Theoretical S = n^alpha
        # Practical floor of 2000 prevents issues with tiny toy graphs
        calculated_s = int(math.pow(args.n_global, args.alpha) * 1000)
        s_edges = max(calculated_s, 2000)

        # R = sqrt(log n)
        r_rounds = max(2, int(math.sqrt(math.log(max(args.n_global, 10)))))

        return cls(
            alpha=args.alpha,
            n_global=args.n_global,
            m_global=args.m_global,
            S_edges=s_edges,
            R_rounds=r_rounds,
            mem_per_cpu_gb=args.mem_per_cpu,
            enable_metrics=args.metrics,
            enable_test_mode=args.test_mode,
            metrics_output_dir=args.metrics_out,
            safety_factor=getattr(args, 'safety_factor', 1.0)
        )