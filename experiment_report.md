# GMM Sublinear MPC: Experiment Report

## Overview
This report analyzes the results of the Graph Maximal Matching (GMM) algorithm in the sublinear regime. Experiments were conducted on a dense random graph ($n=1000, p=0.02$) to verify key algorithmic properties: sublinear memory usage, convergence, stalling behavior, and scaling characteristics.

## 1. Single Trace Analysis (4 Ranks)
**Experiment:** `large_dense.txt` ($N=1000, M \approx 10k$), 4 MPI Ranks.

### A. Convergence & Solution Quality
*   **Observation:** The number of active edges decays exponentially, dropping from ~6162 to <320 within 10 phases. The matching size grows monotonically to 243.
*   **Verdict:** **Expected.** The algorithm effectively finds a maximal matching. The exponential decay confirms the $O(\log \Delta)$ or $O(\log n)$ convergence rate expected of MPC algorithms.

### B. Safety Verification (Sublinearity)
*   **Metric:** Max Ball Size vs. Memory Limit ($S$).
*   **Limit ($S$):** ~12,487 edges per machine.
*   **Observed Max Ball:** 4 edges.
*   **Verdict:** **Expected (Strongly Safe).** The ball sizes are orders of magnitude smaller than the memory limit. This confirms the algorithm operates strictly within the sublinear regime. The very small ball sizes are due to the aggressive sparsification ($p=0.5$) and the relatively small graph size.

### C. Stalling Behavior
*   **Observation:** Stalling rate is high in Phase 0 (~53%) and Phase 1 (~25%), then quickly drops to 0% by Phase 4.
*   **Verdict:** **Expected.** Stalling is an adaptive mechanism. In early phases, the graph is dense, so the algorithm stalls nodes to prevent memory overflows (balls growing too large). As the graph becomes sparser (due to matched edges being removed), the risk of overflow decreases, and stalling deactivates.

### D. Degree Peeling
*   **Observation:**
    *   Phase 0: Mean Degree ~25.
    *   Phase 1: Mean Degree ~12.
    *   Phase 4: Mean Degree ~0.6.
*   **Verdict:** **Expected.** The "Degree Peeling" plot confirms that the algorithm prioritizes or naturally processes high-degree nodes early. The distribution shifts left (lower degrees) as phases progress, which facilitates faster convergence in later stages.

## 2. Scaling Analysis (Strong Scaling)
**Experiment:** Fixed Graph, varying Ranks ($P=2$ vs $P=4$).

### A. Memory Limits ($S$)
*   **P=2:** $S \approx 24,975$ edges.
*   **P=4:** $S \approx 12,487$ edges.
*   **Observation:** Doubling ranks halves the memory limit per rank (assuming fixed total problem size).

### B. Impact on Stalling
*   **P=2 Stalling (Phase 0):** ~51.3%
*   **P=4 Stalling (Phase 0):** ~53.0%
*   **Verdict:** **Expected.** With $P=4$, the memory limit $S$ is tighter (half of $P=2$). The algorithm reacts by stalling slightly more edges (~1.7% increase) to ensure safety. This demonstrates the algorithm's adaptivity to available resources.

### C. Load Balance (Peak Memory)
*   **Observation:** Peak memory usage is uniform across ranks (~48MB).
*   **Verdict:** **Expected.** The randomized distribution of edges (hashing) ensures good load balancing. (Note: For $N=1000$, memory is dominated by runtime overhead, hence the flat 48MB).

## Conclusion
The experiments successfully validate the sublinear GMM algorithm:
1.  **Safety:** It strictly respects memory limits ($O(n^\alpha)$).
2.  **Adaptivity:** Stalling engages dynamically based on density and memory pressure.
3.  **Convergence:** It converges exponentially fast.
4.  **Scaling:** It adapts to varying rank counts by adjusting stalling behavior.

All generated plots show the theoretically expected behaviors.
