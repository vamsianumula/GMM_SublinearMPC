#!/usr/bin/env python3
"""
scripts/plot_metrics.py
Generates plots from metrics_run.json.
"""

import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_convergence(phases, output_dir):
    indices = [p['phase_idx'] for p in phases]
    active = [p['active_edges'] for p in phases]
    matching = [p['matching_size'] for p in phases]
    
    # Cumulative matching
    cum_matching = np.cumsum(matching)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Phase')
    ax1.set_ylabel('Active Edges (Log)', color=color)
    ax1.plot(indices, active, color=color, marker='o', label='Active Edges')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Cumulative Matching Size', color=color)
    ax2.plot(indices, cum_matching, color=color, marker='s', linestyle='--', label='Matching Size')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Convergence: Active Edges vs Matching Growth')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence.png'))
    plt.close()

def plot_ball_stats(phases, output_dir):
    indices = [p['phase_idx'] for p in phases]
    b_max = [p['ball_max'] for p in phases]
    b_mean = [p['ball_mean'] for p in phases]
    b_p95 = [p['ball_p95'] for p in phases]
    
    plt.figure(figsize=(10, 6))
    plt.plot(indices, b_max, label='Max', marker='^', color='red')
    plt.plot(indices, b_p95, label='P95', marker='x', color='orange')
    plt.plot(indices, b_mean, label='Mean', marker='o', color='green')
    
    plt.xlabel('Phase')
    plt.ylabel('Ball Size (Edges)')
    plt.title('Ball Size Statistics (Safety Check)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, 'ball_stats.png'))
    plt.close()

def plot_stalling(phases, output_dir):
    indices = [p['phase_idx'] for p in phases]
    rates = [p['stalling_rate'] * 100 for p in phases] # %
    
    plt.figure(figsize=(10, 6))
    plt.plot(indices, rates, marker='o', color='purple')
    plt.xlabel('Phase')
    plt.ylabel('Stalling Rate (%)')
    plt.title('Stalling Rate per Phase')
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, 'stalling_rate.png'))
    plt.close()

def plot_system(phases, output_dir):
    indices = [p['phase_idx'] for p in phases]
    comm = [p.get('comm_volume_bytes', 0) / 1024 / 1024 for p in phases] # MB
    wait = [p.get('wait_time_seconds', 0) for p in phases]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    ax1.bar(indices, comm, color='teal', alpha=0.7)
    ax1.set_ylabel('Comm Volume (MB)')
    ax1.set_title('Communication Volume (Rank 0)')
    ax1.grid(True, axis='y')
    
    ax2.bar(indices, wait, color='salmon', alpha=0.7)
    ax2.set_ylabel('Wait Time (s)')
    ax2.set_xlabel('Phase')
    ax2.set_title('Wait Time (Rank 0)')
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'system_metrics.png'))
    plt.close()

def plot_degree_stats(phases, output_dir):
    indices = [p['phase_idx'] for p in phases]
    # Handle cases where deg stats might be missing (backward compatibility)
    d_max = [p.get('deg_max', 0) for p in phases]
    d_mean = [p.get('deg_mean', 0) for p in phases]
    d_p95 = [p.get('deg_p95', 0) for p in phases]
    
    plt.figure(figsize=(10, 6))
    plt.plot(indices, d_max, label='Max', marker='^', color='black')
    plt.plot(indices, d_p95, label='P95', marker='x', color='blue')
    plt.plot(indices, d_mean, label='Mean', marker='o', color='green')
    
    plt.xlabel('Phase')
    plt.ylabel('Degree in Sparse Graph')
    plt.title('Sparsification: Degree Statistics')
    plt.legend()
    plt.title('Sparsification: Degree Statistics')
    plt.legend()
    
    # Only use log scale if we have positive values
    if any(v > 0 for v in d_max):
        plt.yscale('log')
        
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, 'degree_stats.png'))
    plt.close()

def plot_sublinearity(data, output_dir):
    phases = data.get('phases', [])
    S_edges = data.get('S_edges', 0)
    
    indices = [p['phase_idx'] for p in phases]
    b_max = [p['ball_max'] for p in phases]
    
    plt.figure(figsize=(10, 6))
    plt.plot(indices, b_max, label='Max Ball Size', marker='^', color='red')
    
    if S_edges > 0:
        plt.axhline(y=S_edges, color='green', linestyle='--', label=f'Sublinear Limit S={S_edges}')
        
    plt.xlabel('Phase')
    plt.ylabel('Edges')
    plt.title('Sublinearity Verification: Ball Size vs Limit')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'sublinearity_check.png'))
    plt.close()

def plot_comm_sublinearity(data, output_dir):
    phases = data.get('phases', [])
    S_edges = data.get('S_edges', 0)
    
    indices = [p['phase_idx'] for p in phases]
    comm_bytes = [p.get('comm_volume_bytes', 0) for p in phases]
    comm_items = [p.get('comm_volume_items', 0) for p in phases]
    
    # Plot 1: Bytes
    plt.figure(figsize=(10, 6))
    plt.plot(indices, comm_bytes, label='Max Comm Volume (Bytes)', marker='D', color='purple')
    
    if S_edges > 0:
        # Limit: O(S) per machine. Let's assume constant factor C=16 (2 ints)
        limit_bytes = S_edges * 16
        plt.axhline(y=limit_bytes, color='green', linestyle='--', label=f'Ref Limit (S * 16 bytes)')
        
    plt.xlabel('Phase')
    plt.ylabel('Bytes Sent (Max Rank)')
    plt.title('Comm Sublinearity: Max Volume vs Limit (Bytes)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'comm_sublinearity_bytes.png'))
    plt.close()
    
    # Plot 2: Items (Edges/Elements)
    plt.figure(figsize=(10, 6))
    plt.plot(indices, comm_items, label='Max Comm Items (Elements)', marker='o', color='blue')
    
    if S_edges > 0:
        # Limit: O(S) items. 
        # Note: Each edge might involve multiple items (u, v, id, etc). 
        # But roughly O(S).
        plt.axhline(y=S_edges * 3, color='green', linestyle='--', label=f'Ref Limit (S * 3 items)')
        
    plt.xlabel('Phase')
    plt.ylabel('Items Sent (Max Rank)')
    plt.title('Comm Sublinearity: Max Volume vs Limit (Items)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'comm_sublinearity_items.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to metrics_run.json")
    parser.add_argument("--out", default=".", help="Output directory for plots")
    args = parser.parse_args()
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        
    data = load_metrics(args.json_file)
    phases = data.get('phases', [])
    
    if not phases:
        print("No phase data found.")
        return

    print(f"Generating plots for {len(phases)} phases in {args.out}...")
    
    plot_convergence(phases, args.out)
    plot_ball_stats(phases, args.out)
    plot_stalling(phases, args.out)
    plot_system(phases, args.out)
    plot_degree_stats(phases, args.out)
    plot_sublinearity(data, args.out)
    plot_comm_sublinearity(data, args.out)
    
    print("Done.")

if __name__ == "__main__":
    main()
