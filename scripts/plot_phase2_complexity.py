import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import math
import random

def generate_random_graph(filename, N, avg_degree=10):
    M = int(N * avg_degree / 2)
    with open(filename, "w") as f:
        for _ in range(M):
            u = random.randint(0, N-1)
            v = random.randint(0, N-1)
            if u != v:
                f.write(f"{u} {v}\n")

def run_experiment(N, alpha=0.5):
    filename = f"random_p2_{N}.txt"
    generate_random_graph(filename, N)
    
    # Standard S_edges
    S_edges = int(math.pow(N, alpha) * 5)
    S_edges = max(S_edges, 20)
    
    cmd = [
        "mpirun", "-n", "4",
        "python3", "-m", "mm_mpc.cli",
        "--input", filename,
        "--n", str(N),
        "--m", str(N*10), # Approx
        "--alpha", str(alpha),
        "--S_edges", str(S_edges),
        "--R_rounds", "2"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + "/src"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        output = result.stdout
        
        # Parse Phase 2 Metrics for Round 0 (Heaviest Round)
        bytes_match = re.search(r"\[Metrics\] Phase2_Round0_Bytes: (\d+)", output)
        edges_match = re.search(r"\[Metrics\] Phase2_Round0_Edges: (\d+)", output)
        
        if bytes_match and edges_match:
            total_bytes = int(bytes_match.group(1))
            total_edges = int(edges_match.group(1))
            if total_edges > 0:
                return total_bytes / total_edges
            else:
                return 0
        else:
            print(f"Warning: No Phase 2 metrics found for N={N}")
            return 0
            
    except subprocess.CalledProcessError as e:
        print(f"Error running N={N}: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        return 0
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def main():
    Ns = [1000, 2000, 5000, 10000]
    avg_bytes_per_edge = []
    
    print("Running Phase 2 Complexity Experiments...")
    for N in Ns:
        print(f"  Testing N={N}...")
        val = run_experiment(N)
        avg_bytes_per_edge.append(val)
        print(f"    -> Avg Bytes/Edge: {val:.2f}")
        
    # Theoretical Limit (S_edges * sizeof(int))
    # S = 5 * N^0.5
    # This is the UPPER BOUND per edge.
    # In practice, balls are smaller on sparse graphs.
    
    theoretical = [5 * (n**0.5) * 8 for n in Ns] # 8 bytes per int
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(Ns, theoretical, 'b-', label='Theoretical Limit (S * 8 bytes)')
    plt.plot(Ns, avg_bytes_per_edge, 'go-', label='Actual Avg Bytes/Edge')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Graph Size (N)')
    plt.ylabel('Avg Communication per Active Edge (Bytes)')
    plt.title('Phase 2 Complexity: Random Sparse Graph')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    output_file = "phase2_complexity_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
