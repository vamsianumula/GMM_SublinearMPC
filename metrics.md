# MPC Maximal Matching Algorithm - Comprehensive Metrics & Plots Specification (v2.1)

This document is the authoritative specification for all metrics and visualizations required for the Strongly Sublinear MPC Maximal Matching implementation. It integrates core performance tracking with rigorous correctness validation.

**Version 2.1 Updates:**
* Added `Approximation Ratio` (Test Mode) and `Estimated Max Degree` metrics.
* Clarified edge consistency checks apply to *active* edges.
* Explicitly added per-rank ball size tracking.
* Relaxed load balancing and communication expectations to realistic HPC bounds.

---

## **Part 1: Metrics Specification**

### **1. Global Algorithmic Metrics**
*Tracks high-level progress and solution quality.*

* **1.1 Matching Size (|M|)**
    * **Description:** Total number of edges added to the matching.
    * **Why Needed:** Primary measure of solution quality.
    * **Expected Behavior:** Monotonically increasing.
    * **Note:** The theoretical bound $|M| \ge |M^*|/2$ cannot be checked in production (as $|M^*|$ is unknown). Use Metric 1.5 for validation.

* **1.2 Active Edges Count**
    * **Description:** Count of edges remaining in the graph (neither matched nor removed).
    * **Why Needed:** Measures convergence speed.
    * **Expected Behavior:** Exponential or near-exponential decay across phases.

* **1.3 Total Phases & Rounds**
    * **Description:** Count of high-level phases and total MPI rounds.
    * **Why Needed:** Verifies the $O(\sqrt{\log \Delta})$ runtime bound.
    * **Expected Behavior:** Small constant or logarithmic relative to max degree.

* **1.4 Unmatched Vertices Count**
    * **Description:** Number of vertices not incident to any matched edge.
    * **Why Needed:** Complementary convergence metric.
    * **Expected Behavior:** Decreases as matching grows.

* **1.5 Approximation Ratio (Test Mode Only)**
    * **Description:** $|M|_{alg} / |M|_{opt}$.
    * **Why Needed:** Verifies solution quality on small graphs or known instances.
    * **Expected Behavior:** $\ge 0.5$ (Maximal Matching guarantee).

* **1.6 Estimated Max Degree ($\Delta_{est}$)**
    * **Description:** The maximum degree observed in the active subgraph at the start of the phase.
    * **Why Needed:** Verifies that the graph is actually "peeling" and becoming sparser.
    * **Expected Behavior:** Should decrease monotonically (or strictly per phase).

---

### **2. Correctness & Invariant Checks (CRITICAL)**
*Sanity checks to catch silent failures and logic errors. Essential for debugging and validation.*

* **2.1 Matching Invariant (Per Phase)**
    * **Metric:** `max_incident_matched_edges_per_vertex` (Global Max).
    * **Why Needed:** Fundamental definition of a matching.
    * **Expected Behavior:** **MUST be 0 or 1.** Any value > 1 is a critical failure.

* **2.2 Maximality Check (End of Run - Test Mode)**
    * **Metric:** `%active_edges_with_both_endpoints_unmatched`.
    * **Why Needed:** Verifies the "maximal" property.
    * **Expected Behavior:** **MUST be 0.**
    * **Implementation:** Gather final state on small graphs; sample random edges on large graphs.

* **2.3 Edge Count Consistency**
    * **Metric:** `abs(sum(active_degree(v)) / 2 - count(active_edges))`.
    * **Why Needed:** Detects CSR corruption or synchronization issues between vertex and edge states.
    * **Expected Behavior:** **MUST be 0** (modulo stalled/removed edges tracking). Both sides must refer to the *active* set.

* **2.4 Symmetric ID Sanity**
    * **Metric:** `symmetric_eid_failure_count` (Sampled).
    * **Why Needed:** Verifies `eid(u, v) == eid(v, u)`.
    * **Expected Behavior:** **MUST be 0.**

---

### **3. Phase-Specific Diagnostics**
*Diagnoses behavior of specific algorithmic steps.*

* **3.1 Sparsification: Degree Estimates**
    * **Metric:** Statistics (Min, Max, Mean, P95, P99) of `deg_in_sparse`.
    * **Why Needed:** Validates sampling; input for stalling logic.
    * **Expected Behavior:** Max close to `T_phase`; distribution shifts lower over time.

* **3.2 Stalling Rate**
    * **Metric:** % of active edges marked `stalled` in current phase.
    * **Why Needed:** Balances progress vs. memory safety.
    * **Expected Behavior:** Low (<20%), dropping to near-zero. High rates = aggressive `T_phase`.

* **3.3 Stalling Rate by Degree Bucket**
    * **Metric:** Stalling rate stratified by node degree (e.g., top 1%, top 10%, others).
    * **Why Needed:** Confirms stalling targets high-degree nodes.
    * **Expected Behavior:** High rate for top bucket, near-zero for others.

* **3.4 Exponentiation: Ball Size Stats**
    * **Metric:** Statistics (Global Max, Mean, P95) of `size(ball[e])`.
    * **Why Needed:** **CRITICAL** for sublinear guarantee.
    * **Expected Behavior:** **Max MUST be <= `S_edges`.**

* **3.5 Exponentiation: Ball Growth Factor**
    * **Metric:** Ratio of ball size at step `i` vs `i-1`.
    * **Why Needed:** Detects explosion in dense regions.
    * **Expected Behavior:** Rapid initial growth, capping at `S_edges` or component size.

* **3.6 Local MIS Selection Rate**
    * **Metric:** % of edges in sparse graph selected for matching.
    * **Why Needed:** Efficiency of local step.
    * **Expected Behavior:** Significant fraction of sparse edges.

---

### **4. Residual Component Analysis (Before Finish)**
*Justifies the "Finish Small Components" strategy.*

* **4.1 Component Count**
    * **Metric:** `num_components_before_finish`.
    * **Why Needed:** Quantifies fragmentation.
    * **Expected Behavior:** Should be high (many small islands).

* **4.2 Component Size Statistics**
    * **Metric:** `max_component_size_edges`, `p95_component_size_edges`.
    * **Why Needed:** Verifies components are small enough for sequential finishing.
    * **Expected Behavior:** Max size << System Memory. P95 should be very small.

---

### **5. System & Resource Metrics**
*Analyzes distributed system health.*

* **5.1 Memory Breakdown (Per Rank)**
    * **Metric:** `mem_edges`, `mem_balls`, `mem_buffers`, `mem_misc` (RSS).
    * **Why Needed:** Pinpoints memory hogs (e.g., is it the graph or the message buffers?).
    * **Expected Behavior:** Balanced. `mem_balls` usually dominates during exponentiation.

* **5.2 Peak Memory Usage**
    * **Metric:** Max RSS per rank.
    * **Why Needed:** Hard hardware constraint.
    * **Expected Behavior:** < `MEM_PER_CPU`.

* **5.3 Communication Volume**
    * **Metric:** Total bytes sent/received per rank/step.
    * **Why Needed:** Bandwidth bottleneck detection.
    * **Expected Behavior:** Per-rank bytes $\approx O(S_{edges})$. (Strict $\le$ is theoretically nice but practically loose due to overheads).

* **5.4 MPI Wait Time / Imbalance**
    * **Metric:** Time in `MPI_Alltoallv`/`Barrier`.
    * **Why Needed:** Detects stragglers.
    * **Expected Behavior:** Low variance.

* **5.5 Max Message Size**
    * **Metric:** Largest single buffer in `Alltoallv`.
    * **Why Needed:** `INT_MAX` overflow prevention.
    * **Expected Behavior:** < 2GB.

* **5.6 Max Ball Size Per Rank**
    * **Metric:** Maximum size of any ball stored on a specific rank.
    * **Why Needed:** Detects local memory spikes hidden by global averages.
    * **Expected Behavior:** Uniformly $\le S_{edges}$. Spikes on specific ranks indicate poor hashing or edge-case topology.

---

### **6. Load Balancing**

* **6.1 Edge/Vertex Skew**
    * **Metric:** Max/Mean ratio for edges and vertices per rank.
    * **Why Needed:** Hash partition validation.
    * **Expected Behavior:** Within a small constant factor (e.g., 1.0 - 1.5). Perfect 1.0 is unrealistic for random hashing.

* **6.2 Heavy-Hitter Skew**
    * **Metric:** Max `deg_in_sparse` contributions processed by one rank.
    * **Why Needed:** CPU bottleneck detection on celebrity nodes.
    * **Expected Behavior:** Manageable under `Delta <= S_edges` assumption.

---

## **Part 2: Plots Specification**

### **1. Convergence & Correctness Visualization**

* **1.1 Active Edges Decay (Log Scale)**
    * **X:** Phase. **Y:** Active Edges (Log).
    * **Why:** Verifies geometric reduction rate per phase.
    * **Expected:** Downward slope. Flattening implies inefficiency.

* **1.2 Matching Growth Profile**
    * **X:** Phase. **Y:** Cumulative |M|.
    * **Why:** Shows work distribution over time.
    * **Expected:** Early steep growth, plateauing later.

* **1.3 Delta Estimate Decay**
    * **X:** Phase. **Y:** Estimated Max Degree ($\Delta$).
    * **Why:** Confirms graph is becoming sparser.
    * **Expected:** Decreasing trend.

* **1.4 Matching Invariant Check**
    * **X:** Phase. **Y:** `max_incident_matched_edges_per_vertex`.
    * **Why:** Visual proof of correctness.
    * **Expected:** Flat line at 1.0 (or 0). Any spike is a bug.

---

### **2. Safety & Sublinearity (The "Memory Guard" Plots)**

* **2.1 Max Ball Size vs. Capacity**
    * **X:** Phase. **Y:** Global Max Ball Size.
    * **Overlay:** Horizontal line at `S_edges`.
    * **Why:** **Proof of sublinearity.**
    * **Expected:** Never crosses the line.

* **2.2 Ball Size Distribution (Violin Plot)**
    * **X:** Phase. **Y:** Ball Size.
    * **Why:** Reveals tail behavior and outliers.
    * **Expected:** Long tails early, tightening later.

* **2.3 Per-Rank Ball Size Heatmap**
    * **X:** Phase. **Y:** Rank ID. **Color:** Max Ball Size.
    * **Why:** Spot hotspots (e.g., "Rank 12 always has monster balls").
    * **Expected:** Uniform color distribution (good load balance).

* **2.4 Peak Memory Usage Per Rank**
    * **Type:** Bar Chart.
    * **X-Axis:** MPI Rank ID.
    * **Y-Axis:** Peak RSS Memory (GB).
    * **Why:** Detects data skew and hardware limits.
    * **Expected Behavior:** Relatively flat horizon.

---

### **3. System Diagnostics**

* **3.1 Compute vs. Wait Time (Stacked Bar)**
    * **X:** Rank ID (for a specific phase). **Y:** Time.
    * **Stacks:** Compute Time, MPI Wait Time.
    * **Why:** Differentiates compute-bound vs. network/imbalance issues.
    * **Expected:** Balanced bars. High wait time = imbalance.

* **3.2 Communication Volume Breakdown**
    * **X:** Phase. **Y:** Bytes Sent.
    * **Stacks:** Step Type (Sparsify, Exponentiate, etc.).
    * **Why:** Identifies bandwidth hogs.
    * **Expected:** Exponentiation usually dominates.

* **3.3 Peak Memory Horizon**
    * **X:** Rank ID. **Y:** Peak RSS.
    * **Why:** Catch OOM risks and skew.
    * **Expected:** Flat horizon.

---

### **4. Algorithmic Mechanics**

* **4.1 Sparsification Degree Shift**
    * **Type:** Multi-Line KDE or Overlaid Histograms.
    * **X-Axis:** `deg_in_sparse` value (Log Scale).
    * **Y-Axis:** Density/Frequency.
    * **Series:** Phase 1, Phase 2, Phase 3...
    * **Why:** Visualizes the "peeling" of the graph.
    * **Expected Behavior:** Distribution mass shifts left (lower degrees).

* **4.2 Degree vs. Stalling Probability**
    * **Type:** Scatter Plot (Sampled edges).
    * **X-Axis:** `deg_in_sparse`.
    * **Y-Axis:** Boolean (Stalled=1, Active=0) or Probability.
    * **Why:** Validates stalling logic.
    * **Expected Behavior:** Sharp step function or sigmoid around `T_phase`.

* **4.3 Residual Component Size Histogram**
    * **Type:** Histogram.
    * **X-Axis:** Component Size (Edges).
    * **Y-Axis:** Count (Log Scale).
    * **Why:** Confirms small components assumption before sequential finish.
    * **Expected Behavior:** Mass concentrated at small sizes.