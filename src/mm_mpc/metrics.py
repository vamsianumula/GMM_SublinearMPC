"""
src/mm_mpc/metrics.py
Metrics subsystem for GMM Sublinear MPC.
"""

import json
import os
import sys
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from mpi4py import MPI

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

@dataclass
class PhaseMetrics:
    phase_idx: int
    active_edges: int
    matching_size: int
    delta_est: int
    
    stalling_rate: float
    stalling_rate_by_bucket: Dict[str, float]
    
    ball_max: int
    ball_mean: float
    ball_p95: float
    
    mis_selection_rate: float
    
    # Degree Stats (Added later, must have defaults)
    deg_min: float = 0.0
    deg_max: float = 0.0
    deg_mean: float = 0.0
    deg_p95: float = 0.0
    deg_hist: Dict[str, int] = field(default_factory=dict) # Bucket -> Count
    
    # Optional / Heavy
    ball_growth_factors: List[float] = field(default_factory=list)
    comm_volume_bytes: int = 0
    comm_volume_items: int = 0
    wait_time_seconds: float = 0.0

@dataclass
class RunMetrics:
    phases: List[PhaseMetrics] = field(default_factory=list)
    total_phases: int = 0
    global_matching_size: int = 0
    unmatched_vertices: int = 0
    max_message_size_bytes: int = 0
    
    # Config Metadata
    S_edges: int = 0
    n_global: int = 0
    
    # Correctness
    max_incident_matched_edges: int = 0
    edge_consistency_error: int = 0
    symmetric_id_failures: int = 0
    
    # Test Mode
    approximation_ratio: Optional[float] = None
    maximality_violation_rate: Optional[float] = None
    
    # System Stats
    peak_memory_per_rank: Dict[int, int] = field(default_factory=dict)

class MetricsTracker:
    """
    Tracks low-level system metrics (communication, time).
    Passed to mpi_helpers.
    """
    def __init__(self):
        self.bytes_sent = 0
        self.bytes_received = 0
        self.items_sent = 0
        self.items_received = 0
        self.max_msg_size = 0
        self.comm_time = 0.0
        
    def record_comm(self, sent_bytes: int, recv_bytes: int, sent_items: int, recv_items: int, max_msg: int, duration: float):
        self.bytes_sent += sent_bytes
        self.bytes_received += recv_bytes
        self.items_sent += sent_items
        self.items_received += recv_items
        self.max_msg_size = max(self.max_msg_size, max_msg)
        self.comm_time += duration
        
    def reset_phase(self):
        self.bytes_sent = 0
        self.bytes_received = 0
        self.items_sent = 0
        self.items_received = 0
        self.comm_time = 0.0
        # We keep max_msg_size cumulative for the run

class MetricsLogger:
    def __init__(self, config, comm: MPI.Comm):
        self.config = config
        self.comm = comm
        self.rank = comm.Get_rank()
        self.run_metrics = RunMetrics()
        self.tracker = MetricsTracker()
        
    def log_phase(self, pm: PhaseMetrics):
        # In a real distributed system, we might want to aggregate some of these 
        # if they are local-only, but PhaseMetrics passed here are assumed 
        # to be already globally aggregated/reduced by the driver.
        self.run_metrics.phases.append(pm)
        
    def finalize_and_dump(self):
        if self.rank == 0:
            output_dir = self.config.metrics_output_dir
            if not output_dir:
                output_dir = "experiments/results/latest"
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. JSON Dump
            json_path = os.path.join(output_dir, "metrics_run.json")
            with open(json_path, "w") as f:
                json.dump(asdict(self.run_metrics), f, indent=2, cls=NumpyEncoder)
            
            # 2. CSV Dump (Phases)
            csv_path = os.path.join(output_dir, "metrics_phases.csv")
            with open(csv_path, "w") as f:
                # Header
                fields = [
                    "phase_idx", "active_edges", "matching_size", "delta_est",
                    "stalling_rate", "ball_max", "ball_mean", "mis_selection_rate",
                    "comm_volume_bytes", "wait_time_seconds"
                ]
                f.write(",".join(fields) + "\n")
                
                for p in self.run_metrics.phases:
                    row = [
                        str(p.phase_idx), str(p.active_edges), str(p.matching_size), str(p.delta_est),
                        f"{p.stalling_rate:.4f}", str(p.ball_max), f"{p.ball_mean:.2f}", 
                        f"{p.mis_selection_rate:.4f}", str(p.comm_volume_bytes), f"{p.wait_time_seconds:.4f}"
                    ]
                    f.write(",".join(row) + "\n")
            
            print(f"[Metrics] Dumped to {output_dir}")
            sys.stdout.flush()
