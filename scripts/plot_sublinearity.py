import subprocess
import numpy as np
import matplotlib.pyplot as plt
import os
import re

def generate_star_graph(filename, N):
    with open(filename, "w") as f:
        # Center is 0, leaves are 1..N-1
        for i in range(1, N):
            f.write(f"0 {i}\n")

def run_experiment(N, alpha=0.5):
    filename = f"star_{N}.txt"
    generate_star_graph(filename, N)
    
    # S_edges should be N^alpha
    # We set S_edges slightly higher to allow for some overhead, but strictly sublinear
    # Actually, we want to see if it STAYS within limit.
    # So we set S_edges = int(N**alpha) * constant
    # Let's set S_edges = int(N**alpha) * 5 to be safe but still sublinear
    S_edges = int(N**alpha) * 5
    # Ensure S_edges is at least something reasonable
    S_edges = max(S_edges, 20)
    
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
            print(f"Warning: No metrics found for N={N}")
            return 0
            
        return max(max_ball_sizes)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running N={N}: {e}")
        print(e.stdout)
        print(e.stderr)
        return 0
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def main():
    Ns = [100, 500, 1000, 2000, 5000, 10000]
    actual_sizes = []
    
    print("Running Sublinearity Experiments...")
    for N in Ns:
        print(f"  Testing N={N}...")
        max_size = run_experiment(N)
        actual_sizes.append(max_size)
        print(f"    -> Max Ball Size: {max_size}")
        
    # Theoretical Limit (approximate S_edges used)
    # We used S = 5 * N^0.5
    theoretical = [5 * (n**0.5) for n in Ns]
    
    # Linear Baseline
    linear = Ns
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(Ns, linear, 'r--', label='Linear (O(N)) - Naive')
    plt.plot(Ns, theoretical, 'b-', label='Theoretical Limit (O(N^0.5))')
    plt.plot(Ns, actual_sizes, 'go-', label='Actual Max Ball Size')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Graph Size (N)')
    plt.ylabel('Max Storage (Edges)')
    plt.title('Sublinearity Verification: Star Graph Memory Scaling')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    output_file = "sublinearity_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
