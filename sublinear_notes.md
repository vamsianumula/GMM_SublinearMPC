# Strongly Sublinear MPC Maximal Matching – Design & Review Notes

**Context:**  
This document distills a review of your current maximal matching implementation plan (proposal + `dev_reference_v2.1`) against the theory of Ghaffari–Uitto–style strongly sublinear MPC algorithms.

It is meant as an **engineering guide for vNext**, focusing on:
- Where your design is already good.
- Where it is *conceptually* faithful but missing theoretical parameterization.
- Where it could silently violate the **strongly sublinear memory** guarantees unless adjusted.

---

## 0. How to use this document

- Treat this as a **spec + checklist** for your next refactor.
- Anywhere you see **MUST**: that’s a correctness or model-violation risk.
- Anywhere you see **SHOULD**: that’s for aligning with the GU proofs and the “strongly sublinear” claim.
- You can annotate this with references to specific functions / files as you code.

---

## 1. Goals & Model Assumptions

### 1.1 Algorithmic goal

Compute a **maximal matching** in an undirected graph \(G = (V, E)\):

- Output \(M \subseteq E\) such that:
  - No two edges in \(M\) share an endpoint. (Matching)
  - No edge in \(E \setminus M\) can be added without violating the matching property. (Maximality)

### 1.2 MPC model constraints

- Number of machines: \(M\).
- Input size: \(n = |V| + |E|\) (or treat \(|E|\) as dominating).
- Per-machine memory: \(S = n^\alpha\), for some fixed \(0 < \alpha < 1\).
- Total memory: \(\tilde{O}(n)\) (or slightly larger, depending on model variant).
- Goal: **Strongly sublinear memory** means every machine has memory \(n^\alpha\) with \(\alpha < 1\), while:
  - Round complexity is \(\tilde{O}(\sqrt{\log \Delta})\) or similar.
  - The algorithm holds for **all graphs** (including adversarial, large-Δ graphs), under the MPC model.

We assume:
- The graph is initially distributed by hashing vertices / edges across machines.
- Communication is synchronous in rounds, with the usual MPC limits.

---

## 2. Theory Snapshot: Ghaffari–Uitto Style Approach

This is **not** a full re-derivation, just the pieces you need to keep in your mental model while coding.

### 2.1 High-level structure

1. **Start from a LOCAL algorithm** for MIS / maximal matching (e.g., randomized greedy).
2. Run it in **phases** of \(R = \Theta(\sqrt{\log \Delta})\) LOCAL rounds each.
3. For each phase:
   - **Sparsify** the graph / line graph:
     - Randomly sample edges / neighbors so that degrees in the *sparsified* graph stay bounded by some function like \(\Delta^{O(\sqrt{\log \Delta})}\).
   - **Stall** dense nodes / edges:
     - If a vertex or edge sees too many sparsified neighbors, you freeze it for that phase.
   - This keeps the **locality volume** (the number of nodes within distance \(R\)) under control.
4. **Graph exponentiation**:
   - In MPC, use a few rounds to gather each node’s \(R\)-hop neighborhood in the sparsified graph onto a single machine.
5. **Local simulation**:
   - Simulate \(R\) rounds of the LOCAL algorithm entirely locally on that neighborhood.
6. Repeat for a small number of phases until remaining components are small.
7. **Finish small components** with a deterministic / simple algorithm using another exponentiation step.

For maximal matching, the algorithm is often derived from MIS on the **line graph** \(L(G)\):

- Nodes of \(L(G)\) are edges of \(G\).
- Two line-graph nodes are adjacent if their corresponding edges in \(G\) share an endpoint.
- MIS in \(L(G)\) = maximal matching in \(G\).

Crucially, you **do not explicitly build** the line graph; you simulate its structure via the original graph.

### 2.2 What the proofs guarantee (informally)

Roughly:

- After sparsification and stalling:
  - Degree in the sparsified graph is small-ish.
  - R-hop neighborhoods have volume bounded by some function like \(\Delta^{O(\sqrt{\log \Delta})}\).
- Per machine:
  - You assign nodes so that
    \[
      (\text{#nodes on machine}) \times (\text{max locality volume}) \leq S = n^\alpha.
    \]
- Therefore:
  - Each machine can store and simulate all needed neighborhoods for its assigned nodes / edges.
  - Number of phases is \(O(\sqrt{\log \Delta})\).
  - After these phases, remaining components are so small that they can be handled with a simple deterministic algorithm.

---

## 3. Summary of Your Current Plan (v2.1)

### 3.1 Core ideas (where you’re already aligned)

- **Global IDs**:
  - Vertices: stable IDs (integers).
  - Edges: symmetric IDs `eid(u, v) = hash64(min(u,v), max(u,v), "eid")`.
- **Ownership**:
  - Edges are assigned to ranks via hashing on `eid`.
  - Vertices are assigned via hashing on vertex ID.
- **Local storage**:
  - CSR-like adjacency for vertices.
  - Arrays of edges, with `id_to_index` maps.
  - No compaction inside a phase (to keep indices stable).
- **Sparsification**:
  - Per phase, edges may participate in the sparsified line graph depending on random sampling (`hash64(eid, phase, seed)` vs threshold).
- **Stalling**:
  - Edges whose degree in the sparsified line graph exceeds a threshold `T_phase` are marked stalled.
- **Exponentiation**:
  - For each non-stalled edge, you grow an \(R\)-hop neighborhood (a “ball”) in the **sparsified line graph**:
    - Stored as `ball_offsets` + `ball_storage` (flat eid arrays, sorted & deduplicated).
- **Local MIS**:
  - Deterministic priorities via `hash64(eid, phase, "priority")`.
  - On each ball, run a greedy MIS over line-graph neighbors.
  - Chosen edges form a matching.
- **Integration & Clean-up**:
  - Mark matched vertices.
  - Remove all edges incident to matched vertices.
- **Finishing**:
  - If remaining edges ≤ some `SMALL_THRESHOLD` (≤ memory on rank 0), gather to rank 0 and finish sequentially.
  - Otherwise, use a distributed finishing algorithm (e.g. deterministic LOCAL/MIS style).

### 3.2 Good properties

- The **matching invariant** is preserved.
- You simulate the **line-graph neighborhood** implicitly via the original graph.
- Exponentiation and MIS logic is consistent with “simulate LOCAL on sparsified structure”.
- You already have a **memory guard** conceptually, and avoid compaction-mid-phase errors.

---

## 4. Main Gaps w.r.t. Strongly Sublinear Guarantees

You asked specifically about being **very critical** on correctness / sublinearity, so here are the three big issues.

### 4.1 Parameters not tied to theory

You currently treat:

- `sampling_params` (probabilities for including edges / neighbors in the sparsified graph).
- `T_phase_params` (thresholds for stalling).
- `R` (radius per phase).
- Phase counts.

as **configuration knobs**.

From correctness standpoint (eventual maximal matching), this is okay.

From a **“we’re implementing the GU theorem”** standpoint:

- The **probabilistic guarantees** (bounded degrees, bounded local volumes, number of surviving components) rely on **specific relationships** between:
  - \(p_i, p'_i\) (sampling probabilities).
  - Degree thresholds for “light”, “good/bad” vertices/edges.
  - \(R\) and phase count.
- Without fixing these according to the proofs, you **cannot claim**:
  - Number of phases = \(O(\sqrt{\log \Delta})\) w.h.p.
  - Memory requirement per machine = \(n^\alpha\) for all graphs.

**Summary:**  
Your plan is *qualitatively aligned* with the theory, but *quantitatively unspecified*. That’s OK for “experimental algorithm,” not OK for a “faithfully implemented theorem.”

---

### 4.2 Per-machine memory vs. storing all balls

You currently plan to:

- For every non-stalled edge \(e\) on a rank, store its entire \(R\)-hop ball in the sparsified line graph.

Let:

- `E_rank` = number of edges assigned to that rank (≈ \(S = n^\alpha\)).
- `B_max` = maximum ball size (max |\(B_R(e)\)| over edges on that rank).

Then **total ball storage** on that rank is roughly:

\[
\text{ball_storage_size} \approx E_\text{rank} \cdot B_\text{max}.
\]

GU’s analysis guarantees something like:

- Per node / edge, \(B_\text{max}\) is small-ish (e.g. \(\Delta^{O(\sqrt{\log \Delta})}\)).
- And they choose assignment so that:
  \[
    (\text{#elements per machine}) \times B_\text{max} \leq S.
  \]

**Your plan’s problem**:

- You assign ~\(S\) edges per rank.
- You then build balls for **all** those edges.
- Even if each *individual* ball is ≤ \(S\), the product \(S \cdot B_\text{max}\) can be \(\gg S\).
- That means: on paper, your per-machine memory is **not guaranteed sublinear** — it could be much larger than \(n^\alpha\) in worst cases.

Relying on a `memory_guard` that **aborts when this happens** is good engineering, but:

- It’s not the same as **proving it never happens** on valid inputs.
- For a worst-case graph (adversarial large-Δ), you might:
  - Blow up memory.
  - Or stall / abort frequently.

**Bottom line:**  
The “store all balls simultaneously” strategy is **not theoretically justified** for strongly sublinear MPC.

---

### 4.3 High-degree vertices & load balancing

Handling very large Δ is delicate:

- GU explicitly discusses that assuming “all edges of a node fit on one machine” can fail when Δ is large.
- They use degree-class tricks and more careful load-balancing in those regimes.

Your current design:

- Uses simple hashing of eids to ranks.
- Uses stalling and sparsification heuristics.
- But **does not**:
  - Stratify by degree.
  - Treat huge-degree vertices specially.
  - Prove that exponentiation and balls for edges near these vertices can be stored with per-machine memory \(≤ n^\alpha\).

In practice, this might work fine on “nice” graphs, but:

- On **worst-case** graphs, this is the regime where:
  - Neighborhoods explode.
  - Your balls become big.
  - Load becomes skewed.

---

## 5. Recommended Design Changes for vNext

This section is the “here’s what to change” part.

### 5.1 Explicit parameter derivation layer

Add a **separate module / section** that:

- Takes as input:
  - \(n\), \(\Delta\) (or an estimate), \(S = n^\alpha\).
- Produces:
  - `R` (radius per phase, e.g. \(\Theta(\sqrt{\log \Delta})\)).
  - `num_phases`.
  - `sampling_params` (probabilities per phase).
  - `T_phase_params` (stall thresholds per phase).
  - Bounds on **expected** ball sizes and degrees in the sparsified line graph.

This module should:

- Encode the theoretical relationships (e.g. light/heavy classifications, “good” vertices, etc.).
- Provide either:
  - Direct formulas (if you reproduce their constants).
  - Or at least **conservative upper bounds** that you know keep ball sizes below some target.

Even if you don’t fully re-prove everything:

- Centralizing parameter logic helps you:
  - Log and adjust them.
  - Experiment while staying within a known envelope.
  - Later, plug in more theoretically justified constants.

**Checklist for this section:**

- [ ] Define \(R = c_R \cdot \sqrt{\log \max(\Delta, 2)}\) for some constant \(c_R\).
- [ ] Define per-phase sampling probabilities \(p_i, p'_i\) as explicit functions of Δ.
- [ ] Define stall threshold `T_phase` as a function of these probs and Δ (e.g. some higher percentile of expected sparsified degree).
- [ ] Document expected bounds on:
  - Degree in sparsified line graph.
  - Ball size for radius \(R\).

---

### 5.2 Batching / throttling ball construction

To make exponentiation **provably** respect per-machine memory \(S\):

**Core idea:**  
You **do not** build balls for *all* edges at once. You only keep balls for a **subset** of edges per batch.

Let:

- `S_edges` ≈ \(n^\alpha\) = edge storage capacity per rank.
- `B_max` = allowed max ball size (upper bound from parameter module).
- Choose `max_active_balls` so that:
  \[
    \text{max_active_balls} \cdot B_\text{max} \leq S_\text{edges} / c
  \]
  for some constant safety factor \(c > 1\).

#### Concrete batching plan

1. On each rank, gather the list of **local non-stalled edges**: `active_edges`.
2. Partition them into batches:
   - Each batch has at most `max_active_balls` edges.
3. For each batch:
   1. Run exponentiation to build balls **only for edges in this batch**.
   2. Run local MIS on these balls; decide which edges in this batch are chosen / removed.
   3. Apply global integration (matched vertices, edge deletions).
   4. Discard the balls for this batch.

4. Proceed to the next batch.

**Consequences:**

- Max ball storage per rank is now:
  \[
    \text{ball_storage_size} \le \text{max_active_balls} \cdot B_\text{max} \le S_\text{edges}/c.
  \]
- You can then add edge arrays, adjacency, etc., and still stay within \(S_\text{edges}\).

**Checklist for batching:**

- [ ] Implement a batching mechanism for non-stalled edges.
- [ ] Add assertions:
  - `len(batch) ≤ max_active_balls`.
  - `estimated_ball_storage ≤ S_edges / c`.
- [ ] Adjust progress logic:
  - Ensure you still do at most `num_phases` phases, but each phase may have several batches.

Batching trades off **more MPC rounds** vs. **stronger memory guarantees**. But for correctness + theory alignment, it’s the right move.

---

### 5.3 High-degree vertices & load-balancing

You should introduce at least **one** of the following strategies:

#### Option A: Explicit Δ upper bound assumption

- Decide to **only claim theory** under a modest assumption, e.g.:

  > For our theoretical analysis, assume global maximum degree  
  > \(\Delta \le n^\beta\) for some \(\beta < \alpha\).

- Then:
  - You can pick parameters ensuring the product:
    \[
      (\text{#edges per rank}) \cdot (\text{ball volume}) \le S_\text{edges}.
    \]
  - You avoid the nastiest high-degree corner cases theoretically.

**This is the simplest option**, but it leaves a hole for real graphs with extremely skewed degrees.

#### Option B: Degree-class decomposition

- In pre-processing:
  - Compute approximate degrees for vertices (MPC-friendly).
  - Define several **degree classes**, e.g.:
    - Class 0: low degree.
    - Class 1: medium.
    - Class 2: high.
    - …
  - Assign edges / vertices to these classes.

- Process degree classes in stages/phases:
  - For very high-degree classes:
    - Use stronger sparsification.
    - More aggressive stalling.
    - Possibly special handling for adjacency distribution.

This is closer to what the theory does in spirit (even if not identical to the proofs).

**Checklist:**

- [ ] Decide whether to:
  - (A) Assume an upper bound on Δ, or
  - (B) Implement degree classes.
- [ ] If (B), define:
  - Degree thresholds.
  - Class-specific sampling / stalling parameters.

---

### 5.4 Finishing small components: tie to theory

Your current heuristic:

- If global active edges ≤ `SMALL_THRESHOLD`, and `SMALL_THRESHOLD ≤ (memory on rank 0)`, then gather to rank 0 and finish sequentially.

To align better with theory:

- The GU analysis gives an upper bound on residual component size after the main randomized phases, something like \(O(\Delta^4 \log n)\) per component.
- You want the gathered graph to fit within rank 0’s memory \(≤ S_\text{edges}\).

**Recommendation:**

- Define:

  ```text
  SMALL_THRESHOLD = min( S_edges / c,  C * Δ^4 * log n )
