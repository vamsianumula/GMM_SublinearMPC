#!/usr/bin/env python3
"""
scripts/plot_suite.py
Generates plots aggregating multiple runs (Scaling, Density Sweeps).
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_suite_metrics(suite_dir):
    """
    Walks through suite_dir, finds metrics_run.json files.
    Returns a list of (run_name, metrics_dict).
    Assumes directory structure: suite_dir/run_name/metrics_run.json
    """
    runs = []
    if not os.path.exists(suite_dir):
        print(f"Suite directory {suite_dir} does not exist.")
        return runs
        
    for d in sorted(os.listdir(suite_dir)):
        path = os.path.join(suite_dir, d)
        if os.path.isdir(path):
            json_path = os.path.join(path, "metrics_run.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    runs.append((d, data))
                except Exception as e:
                    print(f"Failed to load {json_path}: {e}")
    return runs

def parse_run_name(name):
    """
    Tries to extract parameters from run name like 'p4_d10' -> {'p': 4, 'd': 10}.
    """
    params = {}
    parts = name.split('_')
    for p in parts:
        if p.startswith('p') and p[1:].isdigit():
            params['p'] = int(p[1:])
        elif p.startswith('d') and p[1:].isdigit():
            params['d'] = int(p[1:]) # Density or Delta
        elif p.startswith('n') and p[1:].isdigit():
            params['n'] = int(p[1:])
    return params

def plot_scaling_behavior(runs, output_dir):
    """
    Plots metrics vs Number of Ranks (P).
    Expects runs to have 'p' parameter in name.
    """
    # Filter runs that have 'p'
    scaling_runs = []
    for name, data in runs:
        params = parse_run_name(name)
        if 'p' in params:
            scaling_runs.append((params['p'], data))
            
    if not scaling_runs:
        print("No scaling runs found (names must contain p<N>).")
        return
        
    # Sort by P
    scaling_runs.sort(key=lambda x: x[0])
    
    ps = [x[0] for x in scaling_runs]
    
    # Metrics to extract
    stalling_rates = []
    total_phases = []
    max_ball_sizes = []
    peak_mems = []
    
    for _, data in scaling_runs:
        phases = data.get('phases', [])
        
        # Avg Stalling Rate
        rates = [p.get('stalling_rate', 0) for p in phases]
        avg_rate = np.mean(rates) if rates else 0
        stalling_rates.append(avg_rate * 100)
        
        # Total Phases
        total_phases.append(len(phases))
        
        # Max Ball Size (Global Max over run)
        b_max = max([p.get('ball_max', 0) for p in phases]) if phases else 0
        max_ball_sizes.append(b_max)
        
        # Peak Memory (Max of any rank)
        peaks = data.get('peak_memory_per_rank', {})
        if peaks:
            # keys are strings in JSON
            peak_val = max(peaks.values()) / 1024 / 1024 # MB
        else:
            peak_val = 0
        peak_mems.append(peak_val)
        
    # Plot 1: Stalling vs Ranks
    plt.figure(figsize=(8, 5))
    plt.plot(ps, stalling_rates, marker='o', color='purple')
    plt.xlabel('Number of Ranks (P)')
    plt.ylabel('Avg Stalling Rate (%)')
    plt.title('Scaling: Stalling Rate vs Ranks')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'scaling_stalling.png'))
    plt.close()
    
    # Plot 2: Phases vs Ranks
    plt.figure(figsize=(8, 5))
    plt.plot(ps, total_phases, marker='s', color='blue')
    plt.xlabel('Number of Ranks (P)')
    plt.ylabel('Total Phases')
    plt.title('Scaling: Convergence Speed vs Ranks')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'scaling_phases.png'))
    plt.close()
    
    # Plot 3: Max Ball Size vs Ranks
    plt.figure(figsize=(8, 5))
    plt.plot(ps, max_ball_sizes, marker='^', color='red')
    plt.xlabel('Number of Ranks (P)')
    plt.ylabel('Max Ball Size (Edges)')
    plt.title('Scaling: Ball Size Adaptation')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'scaling_ball_size.png'))
    plt.close()
    
    # Plot 4: Peak Memory vs Ranks (Load Balance Check)
    # This is tricky because "Peak Memory" usually decreases as P increases (strong scaling).
    # But we want to see if it's uniform. 
    # Let's plot the Max Peak Memory across ranks.
    plt.figure(figsize=(8, 5))
    plt.plot(ps, peak_mems, marker='D', color='green')
    plt.xlabel('Number of Ranks (P)')
    plt.ylabel('Max Peak Memory (MB)')
    plt.title('Scaling: Peak Memory vs Ranks')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'scaling_memory.png'))
    plt.close()

def plot_load_balance(runs, output_dir):
    """
    Plots Peak Memory per Rank for the largest run found.
    """
    # Find run with max P
    max_p_run = None
    max_p = -1
    
    for name, data in runs:
        params = parse_run_name(name)
        if 'p' in params and params['p'] > max_p:
            max_p = params['p']
            max_p_run = (name, data)
            
    if not max_p_run:
        return
        
    name, data = max_p_run
    peaks = data.get('peak_memory_per_rank', {})
    if not peaks: return
    
    # Sort by rank ID
    ranks = sorted([int(k) for k in peaks.keys()])
    mems = [peaks[str(r)] / 1024 / 1024 for r in ranks] # MB
    
    plt.figure(figsize=(10, 6))
    plt.bar(ranks, mems, color='teal', alpha=0.7)
    plt.xlabel('Rank ID')
    plt.ylabel('Peak Memory (MB)')
    plt.title(f'Load Balance: Peak Memory per Rank ({name})')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'load_balance_memory.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite_dir", help="Directory containing run subdirectories")
    parser.add_argument("--out", default=".", help="Output directory for plots")
    args = parser.parse_args()
    
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        
    runs = load_suite_metrics(args.suite_dir)
    if not runs:
        print("No runs found.")
        return
        
    print(f"Found {len(runs)} runs. Generating suite plots...")
    
    plot_scaling_behavior(runs, args.out)
    plot_load_balance(runs, args.out)
    
    print("Done.")

if __name__ == "__main__":
    main()
