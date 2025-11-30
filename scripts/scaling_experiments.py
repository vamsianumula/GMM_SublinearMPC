#!/usr/bin/env python3
"""
scripts/scaling_experiments.py
Orchestrates scaling experiments for GMM Sublinear MPC.
"""

import os
import subprocess
import json
import time

# Configuration
RANKS = [1, 2, 4]
GRAPHS = [
    {"name": "dense", "file": "dense.txt", "n": 1000, "m": 24798},
    {"name": "star", "file": "star.txt", "n": 1000, "m": 999},
    {"name": "regular", "file": "regular.txt", "n": 1000, "m": 1500},
    {"name": "dense_small", "file": "dense_small.txt", "n": 100, "m": 500},
]
BASE_OUT_DIR = "experiments/scaling"

def run_experiment(graph, ranks):
    print(f"--- Running {graph['name']} on {ranks} ranks ---")
    out_dir = os.path.join(BASE_OUT_DIR, graph['name'], f"{ranks}p")
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        "mpirun", "-n", str(ranks),
        "python3", "-m", "src.mm_mpc.cli",
        "--input", graph['file'],
        "--n", str(graph['n']),
        "--m", str(graph['m']),
        "--metrics-out", out_dir
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    if result.returncode != 0:
        print(f"FAILED: {graph['name']} on {ranks} ranks")
        print(result.stderr)
        return None
        
    print(f"Success ({duration:.2f}s). Output in {out_dir}")
    
    # Add duration to the metrics json manually (since driver doesn't track total wall time)
    json_path = os.path.join(out_dir, "metrics_run.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        data['wall_time_seconds'] = duration
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    return out_dir

def main():
    print("Starting Scaling Experiments...")
    
    # Ensure graphs exist
    if not os.path.exists("dense.txt"):
        print("Generating graphs first...")
        subprocess.run(["./scripts/generate_graphs.py"])
        
    for graph in GRAPHS:
        for r in RANKS:
            run_experiment(graph, r)
            
    print("All experiments completed.")

if __name__ == "__main__":
    main()
