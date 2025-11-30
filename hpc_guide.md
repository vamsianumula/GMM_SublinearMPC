# HPC Scalability Suite Guide

This guide explains how to run the GMM Sublinear MPC scalability experiments on a High-Performance Computing (HPC) cluster.

## 1. The Suite Components

The suite consists of three scripts located in `scripts/`:

1.  **`run_scaling_step.py` (The Worker)**:
    *   **Role**: Runs a single instance of the GMM algorithm for a specific graph size ($N$) and number of ranks ($P$).
    *   **Input**: `--n` (Graph Size), `--out` (Output JSON path).
    *   **Output**: A JSON file containing execution time and success status.
    *   **Logic**: Generates a random graph, runs the algorithm, measures wall-clock time, and dumps metrics.

2.  **`run_cluster_scaling.sh` (The Orchestrator)**:
    *   **Role**: Loops through different rank counts (e.g., 2, 4, 8, 16...), calls `mpirun` for each, and collects results.
    *   **Customization**: You **MUST** edit the `RANKS` array in this script to match your cluster's capabilities (e.g., `RANKS=(32 64 128)`).

3.  **`plot_scaling.py` (The Visualizer)**:
    *   **Role**: Parses the generated JSON files and plots the scaling curve.
    *   **Output**: `scalability_plot.png`.

## 2. How to Run on a Cluster

### Step A: Configure the Orchestrator
Open `scripts/run_cluster_scaling.sh` and update the `RANKS` array:
```bash
# Example for a large cluster
RANKS=(16 32 64 128 256)
```

### Step B: Submit the Job
Depending on your cluster scheduler (SLURM, PBS), you might need to wrap the orchestrator.

**For SLURM (sbatch):**
Create a `submit.slurm` file:
```bash
#!/bin/bash
#SBATCH --job-name=gmm_scaling
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=64
#SBATCH --time=02:00:00

# Load modules (if needed)
module load python/3.8 openmpi

# Install dependencies (if needed)
pip install --user -r requirements.txt

# Run the orchestrator
bash scripts/run_cluster_scaling.sh
```
Then run: `sbatch submit.slurm`

**For Interactive/Shell:**
Simply run:
```bash
bash scripts/run_cluster_scaling.sh
```

## 3. What are we Plotting? (Strong Scaling)

The script generates a **Strong Scaling** plot.

*   **X-Axis**: Number of Processes ($P$).
*   **Y-Axis**: Execution Time (Seconds).
*   **Blue Line (Actual)**: The measured time taken by your algorithm.
*   **Red Dashed Line (Ideal)**: The theoretical perfect scaling curve ($T = T_{base} / P$).

### Interpretation
*   **Good Result**: The Blue line tracks the Red line closely. This means doubling the processors halves the runtime.
*   **Saturation**: Eventually, the Blue line will flatten out. This happens when communication overhead dominates computation. The point where it flattens is your **Scalability Limit**.
*   **Super-Linear**: If Blue is *below* Red, it means you are getting *better* than perfect scaling (rare, usually due to cache effects).

## 4. Validation Checklist
Before running on 1000 cores, verify locally:
1.  [x] `run_scaling_step.py` runs without error for small N.
2.  [x] JSON output is valid.
3.  [x] `plot_scaling.py` generates a PNG.
