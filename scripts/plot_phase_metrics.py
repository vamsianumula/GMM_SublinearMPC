import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import math
import random

def generate_random_graph(filename, N, avg_degree=10):
    M = int(N * avg_degree / 2)
    edges = set()
    with open(filename, "w") as f:
        while len(edges) < M:
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u != v:
                if u > v: u, v = v, u
                if (u, v) not in edges:
                    edges.add((u, v))
                    f.write(f"{u} {v}\n")

def run_experiment(N, alpha=0.5):
    filename = f"random_metrics_{N}.txt"
    generate_random_graph(filename, N)
    
    S_edges = int(math.pow(N, alpha) * 5)
    S_edges = max(S_edges, 20)
    
    cmd = [
        "mpirun", "-n", "4",
        "python3", "-m", "mm_mpc.cli",
        "--input", filename,
        "--n", str(N),
        "--m", str(N*10),
        "--alpha", str(alpha),
        "--S_edges", str(S_edges),
        "--R_rounds", "2"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + "/src"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running N={N}: {e}")
        print(f"Stderr: {e.stderr}")
        return ""
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def parse_metrics(output):
    phases = []
    active_edges = []
    new_matches = []
    max_ball_sizes = []
    phase2_bytes = []
    
    current_phase = -1
    
    # Regex patterns
    p_active = re.compile(r"=== Phase (\d+) \| Active: (\d+) ===")
    p_matches = re.compile(r"\[Metrics\] Phase(\d+)_NewMatches: (\d+)")
    p_ball = re.compile(r"\[Metrics\] Rank \d+ MaxBallSize: (\d+)")
    p_bytes = re.compile(r"\[Metrics\] Phase2_Round(\d+)_Bytes: (\d+)")
    
    # Temp storage for max ball per phase
    phase_max_balls = {}
    
    for line in output.splitlines():
        # Active Edges
        m = p_active.search(line)
        if m:
            phase = int(m.group(1))
            count = int(m.group(2))
            phases.append(phase)
            active_edges.append(count)
            current_phase = phase
            
        # Matches
        m = p_matches.search(line)
        if m:
            count = int(m.group(2))
            new_matches.append(count)
            
        # Max Ball Size (Aggregate across ranks)
        m = p_ball.search(line)
        if m:
            size = int(m.group(1))
            if current_phase not in phase_max_balls:
                phase_max_balls[current_phase] = 0
            phase_max_balls[current_phase] = max(phase_max_balls[current_phase], size)
            
        # Phase 2 Bytes
        m = p_bytes.search(line)
        if m:
            count = int(m.group(2))
            phase2_bytes.append(count)
            
    # Align lists (fill missing with 0 if needed)
    # Assuming sequential phases
    ball_list = [phase_max_balls.get(p, 0) for p in phases]
    
    # Cumulative Matches
    cum_matches = np.cumsum(new_matches)
    
    return phases, active_edges, cum_matches, ball_list, phase2_bytes

def main():
    N = 10000
    print(f"Running Phase Metrics Experiment (N={N})...")
    output = run_experiment(N)
    
    if not output:
        print("Experiment failed.")
        return
        
    phases, active, matches, balls, bytes_p2 = parse_metrics(output)
    
    # Truncate to min length to avoid mismatch
    min_len = min(len(phases), len(active), len(matches), len(balls), len(bytes_p2))
    phases = phases[:min_len]
    active = active[:min_len]
    matches = matches[:min_len]
    balls = balls[:min_len]
    bytes_p2 = bytes_p2[:min_len]
    
    print(f"Parsed {min_len} phases.")
    
    # Plotting 2x2 Grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Convergence (Active Edges)
    axs[0, 0].plot(phases, active, 'b-o')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('Convergence: Active Edges (Log Scale)')
    axs[0, 0].set_xlabel('Phase')
    axs[0, 0].set_ylabel('Active Edges')
    axs[0, 0].grid(True)
    
    # 2. Matching Progress
    axs[0, 1].plot(phases, matches, 'g-o')
    axs[0, 1].set_title('Matching Progress (Cumulative)')
    axs[0, 1].set_xlabel('Phase')
    axs[0, 1].set_ylabel('Total Matches Found')
    axs[0, 1].grid(True)
    
    # 3. Memory Pressure (Max Ball Size)
    axs[1, 0].plot(phases, balls, 'r-o')
    axs[1, 0].set_title('Memory Pressure: Max Ball Size')
    axs[1, 0].set_xlabel('Phase')
    axs[1, 0].set_ylabel('Max Ball Size')
    axs[1, 0].grid(True)
    
    # 4. Bandwidth (Phase 2 Bytes)
    axs[1, 1].plot(phases, bytes_p2, 'm-o')
    axs[1, 1].set_title('Bandwidth: Exponentiation Step')
    axs[1, 1].set_xlabel('Algorithm Round (Phase)')
    axs[1, 1].set_ylabel('Bytes Sent in Exponentiation')
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    output_file = "phase_metrics_dashboard.png"
    plt.savefig(output_file)
    print(f"Dashboard saved to {output_file}")

if __name__ == "__main__":
    main()
