#!/usr/bin/env python3
"""
scripts/run_all_experiments.py
Master script to run ALL experiments and generate ALL plots.
1. Detailed Analysis Runs (4 ranks) -> experiments/results/{graph}
2. Scaling Runs (1, 2, 4 ranks) -> experiments/scaling/{graph}/{ranks}p
3. Plot Generation
4. Artifact Copying
"""

import os
import subprocess
import shutil
import time

GRAPHS = [
    {"name": "dense", "file": "dense.txt", "n": 1000, "m": 24798},
    {"name": "star", "file": "star.txt", "n": 1000, "m": 999},
    {"name": "regular", "file": "regular.txt", "n": 1000, "m": 1500},
    {"name": "dense_small", "file": "dense_small.txt", "n": 100, "m": 500}, # Approx m
]

ARTIFACTS_DIR = "/home/vamsianumula/.gemini/antigravity/brain/e5915768-7f1a-44e0-9b0d-2b38cc59cdaf"

def run_command(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def run_detailed_experiments():
    print("\n=== Running Detailed Analysis Experiments (4 Ranks) ===")
    for g in GRAPHS:
        out_dir = f"experiments/results/{g['name']}"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        
        cmd = [
            "mpirun", "-n", "4",
            "python3", "-m", "src.mm_mpc.cli",
            "--input", g['file'],
            "--n", str(g['n']),
            "--m", str(g['m']),
            "--metrics-out", out_dir
        ]
        run_command(cmd)
        
        # Generate Plots
        plot_out = os.path.join(out_dir, "plots")
        os.makedirs(plot_out, exist_ok=True)
        run_command([
            "./scripts/plot_metrics.py",
            os.path.join(out_dir, "metrics_run.json"),
            "--out", plot_out
        ])
        
        # Copy to Artifacts
        print(f"Copying plots for {g['name']} to artifacts...")
        for f in os.listdir(plot_out):
            if f.endswith(".png"):
                src = os.path.join(plot_out, f)
                dst = os.path.join(ARTIFACTS_DIR, f"{g['name']}_{f}")
                shutil.copy2(src, dst)

def run_scaling_experiments():
    print("\n=== Running Scaling Experiments (1, 2, 4 Ranks) ===")
    # Re-use the existing scaling script logic but call it directly or via subprocess
    # Since scaling_experiments.py is already well-defined, let's just call it.
    run_command(["./scripts/scaling_experiments.py"])
    
    # Generate Scaling Plots
    run_command([
        "./scripts/plot_scaling.py",
        "--out", "experiments/scaling/plots"
    ])
    
    # Copy to Artifacts
    print("Copying scaling plots to artifacts...")
    plot_out = "experiments/scaling/plots"
    for f in os.listdir(plot_out):
        if f.endswith(".png"):
            src = os.path.join(plot_out, f)
            dst = os.path.join(ARTIFACTS_DIR, f"scaling_{f}")
            shutil.copy2(src, dst)

def main():
    start_time = time.time()
    
    # Ensure graphs exist
    if not os.path.exists("dense.txt"):
        run_command(["./scripts/generate_graphs.py"])
        
    run_detailed_experiments()
    run_scaling_experiments()
    
    duration = time.time() - start_time
    print(f"\nAll experiments completed in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
