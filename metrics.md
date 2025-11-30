# MPC Maximal Matching Algorithm - Comprehensive Metrics Specification

This document defines the exhaustive list of metrics required to analyze the performance, correctness, and resource usage of the Strongly Sublinear MPC Maximal Matching algorithm.

## 1. Global Algorithmic Metrics
These metrics track the overall progress and quality of the solution.

### 1.1 Matching Size (|M|)
* **Description:** Total number of edges added to the matching.
* **Why Needed:** Primary measure of solution quality.
* **Expected Behavior:** Should increase monotonically. For maximal matching, |M| >= |M*|/2 (where M* is maximum matching).

### 1.2 Active Edges Count
* **Description:** Number of edges remaining in the graph (not matched and not removed).
* **Why Needed:** Tracks convergence speed.
* **Expected Behavior:** Should decrease exponentially or near-exponentially across phases.

### 1.3 Total Phases & Rounds
* **Description:** Total number of high-level phases and total MPI rounds (including sub-steps).
* **Why Needed:** Verifies the O(sqrt(log Delta)) theoretical runtime bound.
* **Expected Behavior:** Should be small constant or logarithmic relative to max degree.

### 1.4 Unmatched Vertices Count
* **Description:** Number of vertices not incident to any matched edge.
* **Why Needed:** Complementary to matching size; useful for verifying termination condition.
* **Expected Behavior:** Should decrease as matching grows.

---

## 2. Phase-Specific Metrics
These metrics diagnose the behavior of specific algorithmic steps.

### 2.1 Sparsification: Degree Estimates (deg_in_sparse)
* **Description:** Statistics (Min, Max, Mean, P95, P99) of `deg_in_sparse` calculated for each edge.
* **Why Needed:** Validates the sampling process. If `deg_in_sparse` is too high, stalling logic might fail.
* **Expected Behavior:** Max should be close to the phase threshold `T_phase`. Distribution should shift lower in later phases.

### 2.2 Stalling: Stalling Rate
* **Description:** Percentage of active edges marked as `stalled` in the current phase.
* **Why Needed:** Critical for correctness. If too few edges stall, balls will explode in memory. If too many stall, progress slows.
* **Expected Behavior:** Should be low (< 10-20%) but non-zero in early phases for dense graphs.

### 2.3 Exponentiation: Ball Size Statistics
* **Description:** Statistics (Max, Mean, P95) of the number of edges in `ball[e]`.
* **Why Needed:** **CRITICAL** for sublinear guarantee. Monitors the most memory-intensive part of the algorithm.
* **Expected Behavior:** Max ball size MUST ALWAYS be <= `S_edges`.

### 2.4 Exponentiation: Ball Growth Factor
* **Description:** Ratio of ball size at step `i` vs step `i-1`.
* **Why Needed:** Checks if balls are growing too fast (dense regions) or saturating.
* **Expected Behavior:** Should grow rapidly initially but cap at `S_edges` or component size.

### 2.5 Local MIS: Selection Rate
* **Description:** Percentage of edges in the sparse graph selected for the matching in this phase.
* **Why Needed:** Indicates efficiency of the local MIS step.
* **Expected Behavior:** Should be a significant fraction of the sparse graph edges.

---

## 3. System & Resource Metrics (Per-Rank & Aggregate)
These metrics analyze the distributed system performance and bottlenecks.

### 3.1 Peak Memory Usage (Per Rank)
* **Description:** Maximum RAM resident set size (RSS) observed on each MPI rank.
* **Why Needed:** Ensures the algorithm fits in the assigned hardware and validates `n^alpha` scaling.
* **Expected Behavior:** Should never exceed `MEM_PER_CPU` limit. Should be roughly balanced, but `rank 0` might be higher during gather.

### 3.2 Communication Volume: Total Bytes (Per Rank/Step)
* **Description:** Total bytes sent and received by each rank, broken down by step (Sparsify, Exponentiate, etc.).
* **Why Needed:** Identifies bandwidth bottlenecks. MPC assumes communication <= memory.
* **Expected Behavior:** Per-round communication should be <= `S_edges` * `sizeof(edge)`.

### 3.3 MPI Wait Time / Imbalance
* **Description:** Time spent in `MPI_Alltoallv` or `MPI_Barrier` waiting for other ranks.
* **Why Needed:** Detects load imbalance (stragglers).
* **Expected Behavior:** Low variance across ranks. High wait time indicates poor edge/vertex distribution.

### 3.4 Max Message Size
* **Description:** Largest single message buffer sent in `Alltoallv`.
* **Why Needed:** Prevents integer overflow in MPI counts (INT_MAX issue) and validates chunking logic.
* **Expected Behavior:** Must be < 2GB (approx INT_MAX bytes).

---

## 4. Load Balancing Metrics

### 4.1 Edge Distribution Skew
* **Description:** Ratio of (Max Edges on a Rank) / (Mean Edges per Rank).
* **Why Needed:** Validates the initial hash partitioning.
* **Expected Behavior:** Should be close to 1.0 (perfect balance).

### 4.2 Vertex Ownership Skew
* **Description:** Ratio of (Max Vertices Owned by a Rank) / (Mean Vertices per Rank).
* **Why Needed:** Vertex state uses memory too.
* **Expected Behavior:** Close to 1.0.

### 4.3 Heavy-Hitter Skew (Ghost Edges)
* **Description:** Max number of `deg_in_sparse` contributions processed by a single rank.
* **Why Needed:** Identifies if a single rank is owning a celebrity vertex (high degree delta) which could bottleneck CPU even if memory holds.
* **Expected Behavior:** If `Delta <= S_edges` assumption holds, this should be manageable. High skew indicates assumption violation.

# MPC Maximal Matching Algorithm - Comprehensive Plots Specification

This document defines the visualization plan to effectively analyze the MPC Maximal Matching algorithm. These plots are designed to reveal algorithmic convergence, resource bottlenecks, and adherence to theoretical guarantees.

## 1. Algorithmic Convergence & Quality

### 1.1 Active Edges Decay (The "Convergence" Plot)
* **Type:** Line Chart (Log Scale on Y-axis).
* **X-Axis:** Phase Number (1, 2, ...).
* **Y-Axis:** Number of Active Edges (Log Scale).
* **Why:** Verifies the core theoretical claim that the problem size reduces geometrically (or near-geometrically) per phase.
* **Expected Behavior:** A downward sloping line. If it flattens out, the sparsification or MIS step is ineffective.

### 1.2 Matching Growth Profile
* **Type:** Line Chart.
* **X-Axis:** Phase Number.
* **Y-Axis:** Cumulative Matching Size (|M|).
* **Why:** Shows when the "work" is actually being done.
* **Expected Behavior:** Steep growth in early phases (grabbing easy edges), tapering off to a plateau.

### 1.3 Stalling Rate Evolution
* **Type:** Line Chart with Markers.
* **X-Axis:** Phase Number.
* **Y-Axis:** % of Active Edges Marked as "Stalled".
* **Why:** Critical for debugging "frozen" phases.
* **Expected Behavior:** Should be non-zero but manageable (<20%) in early phases for dense graphs, dropping to near-zero in later phases. A spike indicates `T_phase` is too aggressive.

---

## 2. Memory & Sublinear Guarantees (The "Safety" Plots)

### 2.1 Maximum Ball Size vs. Capacity
* **Type:** Line Chart.
* **X-Axis:** Phase Number.
* **Y-Axis:** Size of the *Largest* Ball observed across all ranks.
* **Overlay:** Horizontal dashed line at `S_edges` (Memory Limit).
* **Why:** **The single most important plot for validity.** Proof that the algorithm never violated the sublinear memory constraint.
* **Expected Behavior:** The line must NEVER cross the threshold. It usually spikes in early phases and drops later.

### 2.2 Ball Size Distribution (Violin or Box Plot)
* **Type:** Violin Plot (one violin per phase).
* **X-Axis:** Phase Number.
* **Y-Axis:** Distribution of Ball Sizes.
* **Why:** Averages hide problems. This shows if *most* balls are tiny while a few outliers (heavy hitters) are threatening the memory limit.
* **Expected Behavior:** Long tail distributions in early phases; tightening distributions later.

### 2.3 Peak Memory Usage Per Rank (Load Balance)
* **Type:** Bar Chart.
* **X-Axis:** MPI Rank ID.
* **Y-Axis:** Peak RSS Memory (GB).
* **Why:** Detects data skew. If Rank 0 is 2x higher than others, the finishing phase or partitioning is broken.
* **Expected Behavior:** Relatively flat "horizon". Skews indicate poor hash functions or "ghost edge" concentration.

---

## 3. Communication & System Performance

### 3.1 Communication Volume Breakdown
* **Type:** Stacked Bar Chart.
* **X-Axis:** Phase Number.
* **Y-Axis:** Total Bytes Sent.
* **Stacks:** Step Type (Sparsify, Stalling, Exponentiation, MIS, Integration).
* **Why:** Identifies the bottleneck step. Exponentiation is usually the heaviest.
* **Expected Behavior:** Exponentiation should dominate. If "Sparsify" is huge, the edge-vertex messaging is inefficient.

### 3.2 Phase Runtime Breakdown
* **Type:** Stacked Bar Chart (normalized to 100% or absolute time).
* **X-Axis:** Phase Number.
* **Y-Axis:** Wall-clock Time (seconds).
* **Stacks:** Computation vs. MPI Communication (Wait Time).
* **Why:** Distinguishes between being compute-bound (slow Python/merges) vs. network-bound (bandwidth).
* **Expected Behavior:** Communication time often dominates in MPC. If Computation is huge, check `np.unique` vs `merge_sorted`.

---

## 4. Algorithmic Mechanics (Deep Dive)

### 4.1 Sparsification Degree Shift
* **Type:** Multi-Line Kernel Density Estimate (KDE) or Overlaid Histograms.
* **X-Axis:** `deg_in_sparse` value (Log Scale).
* **Y-Axis:** Density/Frequency.
* **Series:** Phase 1, Phase 2, Phase 3...
* **Why:** Visualizes the "peeling" of the graph.
* **Expected Behavior:** The distribution mass should shift left (lower degrees) with each phase.

### 4.2 Degree vs. Stalling Probability
* **Type:** Scatter Plot (Sampled edges).
* **X-Axis:** `deg_in_sparse`.
* **Y-Axis:** Boolean (Stalled=1, Active=0) or Probability (if using smooth stalling).
* **Why:** Validates that the stalling logic is correctly targeting high-degree nodes.
* **Expected Behavior:** A sharp step function (if hard threshold) or sigmoid (if smooth) around `T_phase`.