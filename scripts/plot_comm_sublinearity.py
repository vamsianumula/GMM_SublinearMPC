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

def run_experiment(N, alpha=0.5):
    filename = f"star_comm_{N}.txt"
    generate_star_graph(filename, N)
    
    # Standard S_edges
    S_edges = int(math.pow(N, alpha) * 5)
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
        
        # Parse TotalCommBytes
        match = re.search(r"\[Metrics\] TotalCommBytes: (\d+)", output)
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: No metrics found for N={N}")
            return 0
            
    except subprocess.CalledProcessError as e:
        print(f"Error running N={N}: {e}")
        return 0
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def main():
    Ns = [1000, 2000, 5000, 10000]
    actual_bytes = []
    
    print("Running Communication Sublinearity Experiments...")
    for N in Ns:
        print(f"  Testing N={N}...")
        total_bytes = run_experiment(N)
        actual_bytes.append(total_bytes)
        print(f"    -> Total Bytes: {total_bytes}")
        
    # Theoretical Limit (Linear O(N) vs Sublinear O(N^alpha))
    # Note: With fixed p=0.5, we expect Linear.
    
    linear = [n * 100 for n in Ns] # Arbitrary constant for visualization
    sublinear = [1000 * (n**0.5) for n in Ns]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(Ns, linear, 'r--', label='Linear (O(N)) - Expected with fixed p')
    plt.plot(Ns, sublinear, 'b-', label='Sublinear (O(N^0.5)) - Goal')
    plt.plot(Ns, actual_bytes, 'go-', label='Actual Total Bytes')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Graph Size (N)')
    plt.ylabel('Total Communication (Bytes)')
    plt.title('Communication Scaling: Star Graph')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    output_file = "comm_sublinearity_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
