# GMM Sublinear MPC: Cluster Reproduction Guide

This guide details how to run the Graph Maximal Matching (GMM) experiments on an HPC cluster and generate the analysis plots.

## 1. Prerequisites

Ensure the following are installed on your cluster nodes:
*   **Python 3.8+**
*   **MPI** (OpenMPI or MPICH)
*   **Python Packages:**
    ```bash
    pip install mpi4py numpy matplotlib networkx
    ```

## 2. Setup

Clone the repository and navigate to the project root:
```bash
cd GMM_SublinearMPC
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 3. Generate Input Graph

Before running experiments, generate the input graph. This is a dense random graph ($N=1000, p=0.02$).

```bash
python3 scripts/generate_graphs.py --type dense --n 1000 --p 0.02 --out experiments/large_dense.txt
```
*Output:* `experiments/large_dense.txt`

## 4. Experiment 1: Single Trace Analysis

This experiment runs the algorithm on 4 ranks to analyze convergence, stalling, and sublinearity safety.

### Running the Experiment
Submit this command to your scheduler (Slurm/PBS) or run interactively:
```bash
mpirun -n 4 python3 scripts/run_single_trace.py
```
*   **Ranks:** 4
*   **Output Directory:** `experiments/results/single_trace`

### Generating Plots
After the run completes, generate the plots:
```bash
python3 scripts/plot_metrics.py experiments/results/single_trace/metrics_run.json --out experiments/results/single_trace
```

**Key Plots to Check:**
*   `sublinearity_check.png`: Max ball size should be well below the green limit line.
*   `stalling_rate.png`: Should start high (~50%) and decay to 0.
*   `degree_peeling.png`: Distribution should shift left over phases.

## 5. Experiment 2: Scaling Analysis

This experiment runs the algorithm with varying rank counts ($P=2, 4$) to verify strong scaling behavior.

### Running the Sweep
Run the following commands (sequentially or in parallel jobs):

**For 2 Ranks:**
```bash
mpirun -n 2 python3 scripts/run_scaling_step.py 2
```

**For 4 Ranks:**
```bash
mpirun -n 4 python3 scripts/run_scaling_step.py 4
```

*   **Output Directory:** `experiments/results/scaling/p2` and `experiments/results/scaling/p4`

### Generating Suite Plots
Aggregate the results from the scaling runs:
```bash
python3 scripts/plot_suite.py experiments/results/scaling --out experiments/results/scaling
```

**Key Plots to Check:**
*   `scaling_stalling.png`: Stalling rate should increase slightly as ranks increase (due to tighter memory limits).
*   `scaling_memory.png`: Peak memory should be roughly uniform (load balanced).

## 6. Results Summary

All results will be located in `experiments/results/`:

*   **Single Trace:** `experiments/results/single_trace/`
*   **Scaling Plots:** `experiments/results/scaling/`
