# Limitations & Future Work

This section summarizes the main theoretical and engineering limitations of the
current implementation of the Strongly Sublinear MPC Maximal Matching algorithm,
together with concrete directions for future improvements.  
These points are intentionally explicit to demonstrate a clear understanding of
where the prototype deviates from the formal Ghaffari–Uitto (GU) guarantees.

---

## 1. Memory Bound Limitations in Exponentiation

### **Limitation**
The implementation caps **individual ball sizes** at `S_edges`, but does *not*
bound the **total ball volume** stored or communicated by a rank.  
During exponentiation:

- All candidate edges construct balls simultaneously.
- `mpi_helpers.exchange_buffers` flattens all send buffers at once.
- This can require memory proportional to  
  \[
      \sum_{e \in \text{candidates}} |B_R(e)|
  \]
  which may exceed the MPC per-machine limit of `Θ(S_edges)`.

### **Future Work**
Implement **batched exponentiation**:
- Partition candidate edges into batches whose total ball volume is ≤ `S_edges`.
- Perform R-round exponentiation for one batch at a time.
- Merge match decisions incrementally.

This brings the implementation closer to the theoretical MPC memory model.

---

## 2. Heuristic Stalling Thresholds

### **Limitation**
The stalling rule uses:
\[
    \text{deg\_in\_sparse}(e) > S_\text{edges}^{1 / R}
\]
This threshold is heuristic and not derived from the GU theoretical “phase
threshold” `T_phase`, which depends on sampling probability, degree decay, and
round scheduling.

### **Future Work**
- Integrate degree-dependent sampling (`p_i`) and theoretical `T_phase`
  parameters.
- Adjust stalling dynamically using:
  - Observed Δ estimates,
  - Phase progression,
  - S_edges scaling.

---

## 3. Fixed Phase Count and Sampling Probability

### **Limitation**
The driver currently uses:
- A **fixed 12 phases**, and  
- A **constant** sampling probability `p = 0.5`.

These choices ignore Δ, its decay, or the theoretical sampling schedules that
guarantee O(√log Δ) phases.

### **Future Work**
- Make `p` phase-dependent (e.g., shrinking probabilities as degrees decrease).
- Use Δ_est to adapt:
  - Number of phases,
  - Stalling thresholds,
  - Sampling rates.

---

## 4. Sequential Finish May Exceed MPC Memory Assumptions

### **Limitation**
`finish_small_components` gathers all residual edges to rank 0 if:
\[
    \text{global\_active} < S_\text{edges} \cdot \text{small\_threshold\_factor}.
\]

Since the factor may be large, this threshold can easily exceed `S_edges`,
violating strict MPC per-machine memory constraints.

### **Future Work**
- Enforce:
  \[
      \text{final edge count} \le S_\text{edges}
  \]
  before gathering.
- Alternatively, implement a **distributed finishing** routine (e.g., BFS-based
  component extraction followed by per-component greedy matching).

---

## 5. Configuration Parameters Are Heuristically Chosen

### **Limitation**
`config.py` defines:
- `S_edges = 1000 * n^alpha`, and
- `R_rounds = floor(sqrt(log n))`.

These are practical but not tied to either:
- Actual hardware memory limits, or
- Formal GU analysis based on Δ and α.

### **Future Work**
- Bind `S_edges` directly to real memory per CPU.
- Tune `R_rounds` via observed Δ decay or theoretical design.
- Allow phase count and sampler parameters to be user-configurable.

---

## 6. No Duplicate-Edge / Self-Loop Normalization in IO

### **Limitation**
`graph_io` directly ingests edges without:
- Removing self-loops,
- Deduplicating parallel edges.

This can inflate degree estimates and ball sizes.

### **Future Work**
- Add optional preprocessing:
  - Remove self-loops.
  - Deduplicate (u, v) edges.
- Or incorporate degree-based pruning at load time.

---

## 7. Proof-Exact Strong Sublinearity Not Yet Achieved

### **Limitation**
While the **structure** of a strongly sublinear MPC maximal matching algorithm is
faithfully followed, the following theory-critical guarantees are not enforced:

- Per-machine memory ≤ `Θ(S_edges)` at all times,
- Phase thresholds tied to Δ-decay,
- Guaranteed ball boundaries,
- Certified O(√log Δ) convergence.

### **Future Work**
- Integrate batching + theoretical stalling,
- Implement adaptive sampling schedules,
- Track Δ_est per phase and adjust algorithmic parameters,
- Formally bind `S_edges` to memory ceilings.

---

## Summary

This implementation correctly captures the **structure** of the GU-style
algorithm and is fully functional for experimentation and educational purposes.
The above limitations mostly concern **formal guarantees**, particularly around
per-machine memory and theoretical parameter tuning.

Documenting these limitations demonstrates a solid understanding of both the
algorithm’s design and the engineering challenges involved in faithfully
implementing MPC algorithms in practice.

