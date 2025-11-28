# Maximal Matching MPC — Developer Reference Document (v2.1)

This is **v2.1**, an updated version incorporating the reviewer’s mandatory fixes **plus the new correction regarding symmetric deterministic edge IDs**.

Changes from v2 → v2.1:
- Added explicit requirement: **eid = hash64(min(u,v), max(u,v), "eid")**
- Added bold WARNING in Section 3.1 about symmetric edge IDs
- No other sections modified unless needed for consistency

---

# 0. Overview

This document is the authoritative implementation specification for the **Strongly Sublinear MPC Maximal Matching** algorithm, based on proposal v6.

It defines:
- Directory structure  
- Modules & APIs  
- All data structures (CSR, ID mappings, ball layouts)  
- Communication & memory rules  
- Global invariants  
- Logging & metrics  
- Testing plan  

This version includes all reviewer-mandated corrections.

---

# 1. Algorithm Summary

The algorithm runs in phases:

1. Sparsification  
2. Stalling edges  
3. Exponentiation (ball-growing)  
4. Local MIS  
5. Integration & deletion  
6. Finishing small components  

Constraints:
- Memory per machine ≤ `S_edges = c * n^α`
- No line-graph materialization
- Fully deterministic hashing
- Cross-rank addressing only via global edge IDs

---

# 2. Repository Structure

```
maximal_matching_mpc/
  README.md
  dev_reference_v2.1.md

  scripts/
    run_slurm.sh
    run_local.sh

  src/mm_mpc/
      config.py
      cli.py
      driver.py
      graph_io.py
      state_layout.py
      partitioning.py

      phases/
        sparsify.py
        stall.py
        exponentiate.py
        local_mis.py
        integrate.py
        finish.py

      utils/
        hashing.py
        mpi_helpers.py
        indexing.py
        monitoring.py
        memory_guard.py
        timers.py

      tests/
        unit/
        integration/
        scale/

  experiments/
    configs/
    results/
    notebooks/
```

---

# 3. Core Data Structures (Revised & Final)

## 3.1 Global ID & Ownership Rules (Corrected)

### Global Edge ID

**Everywhere in the system, the canonical edge ID must be computed as:**

```python
eid = hash64(min(u, v), max(u, v), "eid")
```

### **WARNING (Mandatory):**
**The ID must be symmetric:**  
`eid(u,v) == eid(v,u)`.

If this is violated:
- Two endpoints will compute different IDs
- Vertex aggregation will break
- Ball exponentiation will corrupt
- Matching will become invalid  
**→ Whole algorithm fails**

### Vertex owner
```
owner(v) = hash64(v, "owner") % p
```

### Edge owner
```
edge_owner(eid) = hash64(eid, "edge_owner") % p
```

### Messaging rule
**Only `eid` (global ID) may be used in cross-rank messages.  
Never send local indices across ranks.**

---

## 3.2 EdgeState (Final Revised Structure)

```
edges_local:   (m_local, 2) int64         # endpoints (u,v)
edge_ids:      (m_local,) int64           # global eids (symmetric)

active_mask:   (m_local,) bool
deg_in_sparse: (m_local,) int32
stalled:       (m_local,) bool
priority:      (m_local,) uint64
matched_edge:  (m_local,) bool

# Ball structure (eid-based, CSR-like)
ball_offsets:  (m_local+1,) int64
ball_storage:  flat int64 array of eids   # sorted, unique

# Global → local
id_to_index: Dict[int, int]               # Must stay consistent
```

- All adjacency or ball operations use global **eids**.
- Local indices are internal only and hidden behind `id_to_index`.

---

## 3.3 VertexState (CSR, mandatory)

```
vertex_ids:         (n_local,) int64

vertex_id_to_row:   Dict[int, int]     # small dict

adj_offsets:        (n_local+1,) int64
adj_storage:        (total_incident_edges,) int32
```

No Python lists or dict-of-lists allowed.

---

# 4. Control Flow

```
cli.main
  → driver.run_mpc_matching
      → graph_io.load_and_distribute_graph
      → state_layout.initialize_edge_state
      → state_layout.initialize_vertex_state

      → For phase in 1..P:
            sparsify.compute_phase_participation
            sparsify.compute_deg_in_sparse
            stall.apply_stalling
            exponentiate.build_balls
            local_mis.assign_priorities
            local_mis.run_greedy_mis
            integrate.update_matching_and_prune

      → finish.finish_small_components
      → driver.gather_and_write_results
```

---

# 5. Module Specifications

(Same as v2, except updated to reflect symmetric eid)

## 5.1 config.py  
Contains:
```
alpha
S_edges
R
T_phase_params
sampling_params
small_threshold_factor
memory_soft_threshold = 0.75
memory_hard_threshold = 0.90
```

---

## 5.2 state_layout.py

### initialize_edge_state(...)
- Computes CSR adjacency
- Computes `eid = hash64(min(u,v), max(u,v), "eid")`
- Builds `id_to_index`

### compact_edges_if_needed()
- Trigger only if utilization < threshold
- Rebuild both id_to_index and CSR adjacency
- Never inside a phase step

---

# 6. Sparsification

### participation:
```
include[e,i] = (hash64(eid, phase, i, "sample") < threshold)
```

### deg_in_sparse:
`deg_L_sparse(e) = (d_u - 1) + (d_v - 1)`

---

# 7. Stalling

`stalled[e] = deg_in_sparse[e] > T_phase`

---

# 8. Exponentiation (Corrected)

- Balls store **eids only**
- All communication chunked
- Each step ensures:
  - sorted unique ball
  - size ≤ S_edges
  - memory guard obeyed

---

# 9. Local MIS

### Priority:
```
priority[e] = hash64(eid, phase, "priority")    # uint64
```

### MIS condition:
- No chosen edges share endpoint
- No chosen edges appear in each other's ball

---

# 10. Integration & Deletion

- Mark matched vertices
- Allreduce OR
- Remove edges via `active_mask=False`
- Do not compact mid-phase

---

# 11. Finish Small Components

- If global edges ≤ threshold: gather + sequential solving
- Else: distributed finishing

---

# 12. Invariants

Same as v2, but add:

### Symmetric Edge ID Invariant
**Everywhere in the program:**
```
eid(u, v) == eid(v, u)
```

If false, program correctness collapses.

---

# 13. Memory Guard (Revised)

- Soft: 75% → GC + warning
- Hard: 90% → fail-fast

---

# 14. Metrics & Logging

Same as v2.

---

# 15. Testing

Same as v2:
- Unit tests for symmetric eid
- Correctness tests for mapping consistency

---

# End of Document
