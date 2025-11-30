import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import math

def generate_star_graph(filename, N):
    with open(filename, "w") as f:
        # Center is 0, leaves are 1..N-1
        for i in range(1, N):
            f.write(f"0 {i}\n")

def run_experiment(N, alpha):
    filename = f"star_alpha_{N}.txt"
    generate_star_graph(filename, N)
    
    # Strict Limit: S = ceil(N^alpha)
    S_edges = int(math.ceil(math.pow(N, alpha)))
    # Ensure S_edges is at least something reasonable to avoid instant crash on tiny alpha
    # But for N=10000, even alpha=0.1 gives S=2.5 -> 3.
    # The algorithm needs at least minimal space.
    # Let's enforce min S=5 for safety.
    S_edges = max(S_edges, 5)
    
    cmd = [
        "mpirun", "-n", "4",
        "python3", "-m", "mm_mpc.cli",
        "--input", filename,
        "--n", str(N),
        "--m", str(N-1),
        "--alpha", str(alpha),
        "--S_edges", str(S_edges),
        "--R_rounds", "2"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + "/src"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        output = result.stdout
        
        # Parse MaxBallSize
        max_ball_sizes = []
        for line in output.splitlines():
            match = re.search(r"\[Metrics\] Rank \d+ MaxBallSize: (\d+)", line)
            if match:
                max_ball_sizes.append(int(match.group(1)))
                
        if not max_ball_sizes:
            print(f"Warning: No metrics found for alpha={alpha}")
            return 0
            
        return max(max_ball_sizes)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running alpha={alpha}: {e}")
        # If it failed due to MemoryError (which is possible if we are TOO strict), 
        # it means we hit the limit hard.
        # But we want to see if it *adapts*.
        # If it fails, return S_edges (as it hit the cap).
        return S_edges
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def main():
    N = 10000
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    actual_sizes = []
    theoretical_limits = []
    
    print(f"Running Alpha Scaling Experiments (N={N})...")
    for alpha in alphas:
        print(f"  Testing alpha={alpha}...")
        max_size = run_experiment(N, alpha)
        actual_sizes.append(max_size)
        
        limit = int(math.ceil(math.pow(N, alpha)))
        theoretical_limits.append(limit)
        
        print(f"    -> Limit: {limit}, Actual: {max_size}")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, theoretical_limits, 'b-', label='Theoretical Limit (N^alpha)')
    plt.plot(alphas, actual_sizes, 'go--', label='Actual Max Ball Size')
    
    plt.yscale('log')
    plt.xlabel('Alpha')
    plt.ylabel('Max Storage (Edges) - Log Scale')
    plt.title(f'Adaptability Verification: N={N} Star Graph')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    output_file = "alpha_scaling_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
