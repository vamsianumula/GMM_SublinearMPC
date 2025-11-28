# Maximal Matching MPC — Developer Reference Document (v2)

This is **v2**, updated according to the reviewer's mandatory corrections:
- Stable addressing (no compaction inside loop)
- CSR-based VertexState adjacency
- Global ID → Local Index lookup (`id_to_index`)
- Priority stored as uint64
- Chunked MPI communication
- Improved memory-guard rules

---

# 0. Overview

This document specifies the **full implementation plan** for the Strongly Sublinear MPC Maximal Matching algorithm, consistent with the proposal document v6 and incorporating all mandatory structural fixes.

It defines:

- Folder layout  
- Modules & APIs  
- Core data structures  
- Global invariants  
- Communication patterns  
- Memory guard logic  
- Metrics & logging  
- Testing plan  

This is the reference document developers must follow.

---

# 1. Algorithm Overview

The algorithm consists of the following phases:

1. **Sparsification**
2. **Stalling edges**
3. **Exponentiation (ball-growing)**
4. **Local MIS on balls**
5. **Integration & deletion**
6. **Finish small components**

Constraints:
- Per-machine memory: `S_edges = c * n^α`
- No explicit line-graph representation
- Deterministic hashing (`hash64`)
- All cross-rank addressing uses **global edge IDs**, never local indices
- Communication per-round ≤ S_edges

---

# 2. Repository Structure

```
maximal_matching_mpc/
  README.md
  dev_reference_v2.md

  scripts/
    run_slurm.sh
    run_local.sh

  src/mm_mpc/
      __init__.py
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

# 3. Core Data Structures (Mandatory Revised Version)

## 3.1 Global IDs & Ownership Rules

**Edge ID**  
`eid = deterministic_id(u, v)` (64‑bit int)  
Stable throughout entire algorithm.

**Vertex owner:**  
`owner(v) = hash(v) % p`

**Edge owner:**  
`edge_owner(eid) = hash(eid) % p`  
Ensures stable location as long as global ID is stable.

All cross-rank messages refer to edges by **eid**, not local index.

---

# 3.2 EdgeState (Revised)

Stored on each rank:

```
edges_local:      (m_local, 2) int64         # (u, v)
edge_ids:         (m_local,) int64           # eid for each edge
active_mask:      (m_local,) bool

deg_in_sparse:    (m_local,) int32
stalled:          (m_local,) bool
priority:         (m_local,) uint64          # IMPORTANT: no float64
matched_edge:     (m_local,) bool

# Ball structure (CSR-like)
ball_offsets:     (m_local+1,) int64
ball_storage:     flat int64 array of eids   # NOT local indices

# CRITICAL: Global ID -> Local Index map
id_to_index: Dict[int, int]
```

Notes:
- `id_to_index` maps **global eid → local array index**.
- Must be rebuilt every time edges are compacted.
- Only dictionary allowed in hot path.

---

# 3.3 VertexState (Revised CSR Format)

Old version (dict[list]) is **forbidden**.

New version:

```
vertex_ids:     (n_local,) int64             # row order
vertex_id_to_row: Dict[int, int]             # maps global vid -> CSR row

# CSR adjacency: row = vertex, values = local edge indices
adj_offsets:    (n_local+1,) int64
adj_storage:    (total_incident_edges,) int32
```

Advantages:
- No Python objects in adjacency
- Deterministic memory usage
- Fast scanning of adjacency lists

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

## 5.1 config.py  
Defines parameters:

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

### initialize_edge_state(edges_local, config)
- Creates `EdgeState`
- Builds `id_to_index`

### initialize_vertex_state(comm, edges_local)
- Builds CSR adjacency

### compact_edges_if_needed()
- Only triggered if active utilization < threshold (e.g., < 40%)
- Rebuild `id_to_index` and CSR adjacency
- **Never run inside exponentiation or MIS of a phase**

---

## 5.3 sparsify.py

### compute_phase_participation(edge_state, phase, config)
Uses:
```
include[e,i] = (hash64(eid, phase, i, "sample") % (2^64)) < threshold
```

### compute_deg_in_sparse(edge_state, vertex_state)
- Edges send their eid to vertices
- Vertices count participating edges
- Return `(d_u - 1) + (d_v - 1)`

---

## 5.4 stall.py

```
stalled[e] = (deg_in_sparse[e] > T_phase(phase))
```

---

## 5.5 exponentiate.py

### build_balls(edge_state, vertex_state)

**CRITICAL CHANGES:**
- balls store **global edge IDs**  
- never store local indices
- chunked MPI communication

### Communication Pattern (Chunked)

Each step:
- Partition send buffers so each message block ≤ CHUNK_LIMIT (configurable)
- Use `Alltoallv` on each chunk sequentially

### Ball invariants:
- Sorted & unique
- ball_size ≤ S_edges
- Uses uint64 eids

---

## 5.6 local_mis.py

### assign_priorities()
```
priority[e] = hash64(eid, phase, "priority")    # uint64
```

### run_greedy_mis()
- Uses only `eid`
- Check neighbors via ball_storage

---

## 5.7 integrate.py

- Mark matched vertices
- Allreduce OR on matched vertices
- Delete incident edges by setting `active_mask = False`
- Do **not** compact mid-phase

---

## 5.8 finish.py

- Compute global edge count
- If ≤ SMALL_THRESHOLD:
  - Gather edges to rank 0
  - Run sequential greedy matching
- Else:
  - Distributed LOCAL-style finishing

---

# 6. Correctness Invariants

## Global
- All cross-rank edge references use **eid**
- Matching is valid after each phase
- No line-graph adjacency materialized

## Per-Phase
- deg_in_sparse correct
- stalled flags correct
- ball arrays sorted, unique, ≤ S_edges
- MIS independent set property
- After integration: no edge touches matched vertex

---

# 7. Memory Guard Policy (Revised)

## Soft Threshold (75%)
- Trigger:
  - Log warning
  - Call `gc.collect()`
  - Dump metrics

## Hard Threshold (90%)
- Log emergency
- Abort with safe shutdown
- Dump state to debug file

---

# 8. Logging & Metrics

### Algorithmic
- matching size |M|
- active edges
- deg_in_sparse distribution
- stalling rate
- ball size distribution
- shrinkage per phase

### System
- Per-rank memory
- Communication volume
- Per-phase time
- Peak ball size
- Load balance metrics

Output:
```
experiments/results/<run_id>/metrics.jsonl
```

---

# 9. Testing Plan

## Unit Tests
- hash64 deterministic  
- id_to_index correctness  
- CSR adjacency correctness  
- merge_sorted_unique correctness  

## Integration Tests
- 1–4 ranks on small graphs
- Validate matching with NetworkX

## Scale Tests
- 16–512 ranks  
- Random graphs  
- Power-law graphs  
- Ensure ball <= S_edges always

---

# 10. Debugging Playbook (Updated)

- **Ball size spike** → T_phase too high or sampling too dense  
- **Rank 0 slowdown** → finishing should be distributed  
- **Alltoallv large** → chunk sizes too big or no batching  
- **Memory spike** → CSR adjacency not built correctly OR compaction run too late  

---

# End of Document
