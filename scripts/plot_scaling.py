import json
import os
import argparse
import matplotlib.pyplot as plt
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing result_*.json files")
    args = parser.parse_args()
    
    json_files = glob.glob(os.path.join(args.dir, "result_*.json"))
    
    ranks = []
    durations = []
    
    for fpath in json_files:
        with open(fpath, "r") as f:
            data = json.load(f)
            if data.get("success", False):
                ranks.append(data["ranks"])
                durations.append(data["max_duration"])
                
    if not ranks:
        print("No valid results found.")
        return
        
    # Sort by ranks
    zipped = sorted(zip(ranks, durations))
    ranks, durations = zip(*zipped)
    
    print(f"Plotting results for ranks: {ranks}")
    print(f"Durations: {durations}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, durations, 'b-o', label='Execution Time')
    
    # Ideal Scaling (Strong Scaling: T = T1 / P)
    # We use the first data point as baseline
    base_p = ranks[0]
    base_t = durations[0]
    ideal = [base_t * (base_p / p) for p in ranks]
    
    plt.plot(ranks, ideal, 'r--', label='Ideal Strong Scaling (1/P)')
    
    plt.xlabel('Number of Ranks (P)')
    plt.ylabel('Execution Time (s)')
    plt.title('Strong Scalability Analysis')
    plt.legend()
    plt.grid(True)
    
    output_file = "scalability_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
