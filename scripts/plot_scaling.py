#!/usr/bin/env python3
"""
scripts/plot_scaling.py
Generates aggregate plots from scaling experiments.
"""

import json
import os
import matplotlib.pyplot as plt
import argparse

BASE_OUT_DIR = "experiments/scaling"
GRAPHS = ["dense", "star", "regular"]
RANKS = [1, 2, 4]

def load_data():
    results = {g: {} for g in GRAPHS}
    for g in GRAPHS:
        for r in RANKS:
            path = os.path.join(BASE_OUT_DIR, g, f"{r}p", "metrics_run.json")
            if os.path.exists(path):
                with open(path, 'r') as f:
                    results[g][r] = json.load(f)
    return results

def plot_time(results, output_dir):
    plt.figure(figsize=(10, 6))
    for g in GRAPHS:
        times = []
        valid_ranks = []
        for r in RANKS:
            if r in results[g]:
                times.append(results[g][r].get('wall_time_seconds', 0))
                valid_ranks.append(r)
        if times:
            plt.plot(valid_ranks, times, marker='o', label=g)
            
    plt.xlabel('MPI Ranks')
    plt.ylabel('Wall Time (s)')
    plt.title('Strong Scaling: Execution Time vs Ranks')
    plt.legend()
    plt.grid(True)
    plt.xticks(RANKS)
    plt.savefig(os.path.join(output_dir, 'scaling_time.png'))
    plt.close()

def plot_comm(results, output_dir):
    plt.figure(figsize=(10, 6))
    for g in GRAPHS:
        comm = []
        valid_ranks = []
        for r in RANKS:
            if r in results[g]:
                # Sum comm volume from all phases
                phases = results[g][r].get('phases', [])
                total_bytes = sum(p.get('comm_volume_bytes', 0) for p in phases)
                comm.append(total_bytes / 1024 / 1024) # MB
                valid_ranks.append(r)
        if comm:
            plt.plot(valid_ranks, comm, marker='s', linestyle='--', label=g)
            
    plt.xlabel('MPI Ranks')
    plt.ylabel('Total Comm Volume (MB)')
    plt.title('Communication Overhead vs Ranks')
    plt.legend()
    plt.grid(True)
    plt.xticks(RANKS)
    plt.savefig(os.path.join(output_dir, 'scaling_comm.png'))
    plt.close()

def plot_max_ball(results, output_dir):
    plt.figure(figsize=(10, 6))
    for g in GRAPHS:
        balls = []
        valid_ranks = []
        for r in RANKS:
            if r in results[g]:
                phases = results[g][r].get('phases', [])
                # Max ball size across ALL phases
                max_b = max((p.get('ball_max', 0) for p in phases), default=0)
                balls.append(max_b)
                valid_ranks.append(r)
        if balls:
            plt.plot(valid_ranks, balls, marker='^', label=g)
            
    plt.xlabel('MPI Ranks')
    plt.ylabel('Max Ball Size (Edges)')
    plt.title('Sublinearity Robustness: Max Ball Size vs Ranks')
    plt.legend()
    plt.grid(True)
    plt.xticks(RANKS)
    plt.savefig(os.path.join(output_dir, 'scaling_ball_size.png'))
    plt.close()

def plot_max_comm_per_rank(results, output_dir):
    plt.figure(figsize=(10, 6))
    for g in GRAPHS:
        max_comm = []
        valid_ranks = []
        limit_bytes = 0
        
        for r in RANKS:
            if r in results[g]:
                data = results[g][r]
                phases = data.get('phases', [])
                S_edges = data.get('S_edges', 0)
                
                # S_edges is constant for a graph type, capture it
                if S_edges > 0:
                    limit_bytes = S_edges * 16 # Approx 16 bytes per edge
                
                # comm_volume_bytes in phase metrics is already the Global Max (per phase)
                # We want the peak load any rank experienced in any phase
                peak_load = max((p.get('comm_volume_bytes', 0) for p in phases), default=0)
                max_comm.append(peak_load)
                valid_ranks.append(r)
                
        if max_comm:
            plt.plot(valid_ranks, max_comm, marker='D', label=f'{g} (Peak Load)')
            
        # Plot limit line only once per graph type (or just once if they are similar)
        # For simplicity, let's just plot the limit for the last graph processed if available
        if limit_bytes > 0 and g == GRAPHS[-1]: 
             plt.axhline(y=limit_bytes, color='green', linestyle='--', label=f'Sublinear Limit O(S)')

    plt.xlabel('MPI Ranks')
    plt.ylabel('Peak Comm Volume (Bytes)')
    plt.title('Sublinearity: Peak Per-Rank Communication vs Ranks')
    plt.legend()
    plt.grid(True)
    plt.xticks(RANKS)
    plt.savefig(os.path.join(output_dir, 'scaling_max_comm.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="experiments/scaling/plots", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        
    print("Loading data...")
    results = load_data()
    
    print(f"Generating scaling plots in {args.out}...")
    plot_time(results, args.out)
    plot_comm(results, args.out)
    plot_max_ball(results, args.out)
    plot_max_comm_per_rank(results, args.out)
    print("Done.")

if __name__ == "__main__":
    main()
