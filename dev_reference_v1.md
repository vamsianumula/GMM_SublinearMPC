# Maximal Matching MPC — Developer Reference Document (v1)

## 0. Overview

This document is the complete developer reference for implementing the **Strongly Sublinear MPC Maximal Matching Algorithm** based on the proposal document (v6).  
It defines the directory structure, modules, classes, functions, invariants, communication patterns, logging, metrics, and testing requirements for a fully correct and debuggable implementation in Python + mpi4py.

---

## 1. Project Overview

The goal is to implement the MPC maximal matching algorithm with the following core phases:

1. Sparsification (implicit line-graph sampling)
2. Stalling of dense edges
3. Graph exponentiation (ball-growing)
4. Local MIS on balls
5. Integration and deletion of incident edges
6. Finishing small components

The implementation must strictly respect MPC constraints:
- Per-machine memory S_edges = c * n^α
- Total memory ≤ O(n + m)
- Communication ≤ S_edges per round
- No explicit line graph
- Deterministic hashing instead of randomness

---

## 2. Directory & File Structure

```
maximal_matching_mpc/
  README.md
  dev_reference_v1.md
  scripts/
    run_slurm.sh
    run_local.sh
  src/
    mm_mpc/
      __init__.py
      config.py
      cli.py
      driver.py
      graph_io.py
      partitioning.py
      state_layout.py

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

## 3. Core Data Structures

### 3.1 Global ID & Ownership Rules

- Vertex owner:  
  `owner(v) = hash(v) % p`
- Edge owner:  
  `edge_owner(u,v) = hash(min(u,v), max(u,v)) % p`
- All hashing uses deterministic 64-bit hash `hash64`.

### 3.2 EdgeState

Stored per rank:

```
edges_local: (m_local, 2) int64
active_mask: (m_local,) bool
deg_in_sparse: (m_local,) int32
stalled: (m_local,) bool
priority: (m_local,) float64
ball_offsets: (m_local+1,) int64
ball_storage: flat int64 array
matched_edge: (m_local,) bool
```

### 3.3 VertexState

Stored only for vertices owned on each rank:

```
vertex_ids: (n_local,) int64
adjacency: dict[v] = list(local edge indices)
d_v_sparsify: (n_local,) int32
matched_vertex: (n_local,) bool
```

---

## 4. Control Flow Overview

```
cli.main
  → driver.run_mpc_matching
      → graph_io.load_and_distribute_graph
      → state_layout.init_edge_state
      → state_layout.init_vertex_state

      → for phase in 1..P:
            sparsify.compute_phase_participation
            sparsify.compute_deg_in_sparse
            stall.apply_stalling
            exponentiate.build_balls
            local_mis.assign_priorities
            local_mis.run_greedy_mis
            integrate.update_matching_and_prune

      → finish.finish_small_components
      → driver.gather_and_write_result
```

---

## 5. Module-Level Specifications

### 5.1 config.py

Defines:

```
class Config:
    alpha
    S_edges
    R
    T_phase_params
    sampling_params
    small_threshold_factor
```

### 5.2 graph_io.py

Functions:

- `read_edge_list(path) -> np.ndarray[(m,2)]`
- `distribute_edges(comm, edges) -> edges_local`

### 5.3 state_layout.py

- `init_edge_state(edges_local, config)`
- `init_vertex_state(comm, edges_local)`
- `compact_active_edges(edge_state, vertex_state)`

### 5.4 sparsify.py

- `compute_phase_participation(edge_state, phase, config)`
- `compute_deg_in_sparse(edge_state, vertex_state, phase)`

### 5.5 stall.py

- `apply_stalling(edge_state, phase, config)`

### 5.6 exponentiate.py

- `build_balls(edge_state, vertex_state, config)`
- Internal helpers:
  - `ball_to_vertex_comm`
  - `vertex_expand_balls`
  - `vertex_to_ball_comm`
  - `merge_sorted_unique`

### 5.7 local_mis.py

- `assign_priorities(edge_state, phase)`
- `run_greedy_mis(edge_state, ball_state) -> chosen_edges_local`

### 5.8 integrate.py

- `update_matching_and_prune(edge_state, vertex_state, chosen_edges, comm)`

### 5.9 finish.py

- `finish_small_components(edge_state, vertex_state, config, comm)`

---

## 6. Correctness Invariants

### Global invariants:
- Matching is a valid matching at all times.
- No explicit line-graph adjacency.
- All arrays index-aligned.

### Phase invariants:
- Sparsify: deg_in_sparse[e] = (d_u−1)+(d_v−1)
- Stalling: stalled[e] ↔ deg_in_sparse[e] > T_phase
- Exponentiation:
  - ball[e] sorted & unique
  - size(ball[e]) ≤ S_edges
- MIS:
  - No two chosen edges share an endpoint
- Integration:
  - No remaining active edge touches a matched vertex

---

## 7. Logging & Metrics

### Algorithmic metrics:
- |M| so far
- Active edges count
- deg_in_sparse stats: mean, max, p95
- Ball size distribution
- Stalling fraction
- Per-phase shrinkage

### System metrics:
- Peak memory per rank
- Bytes sent/received
- Alltoallv runtime
- Phase runtimes
- Load balance (edges per rank)

### Output format:
- Rank 0 writes JSON lines to:
  `experiments/results/run_id/metrics.jsonl`

---

## 8. Memory Guard & Fail-Fast Rules

- Abort if:
  - ball[e] > S_edges
  - memory > MEM_PER_CPU * 0.90
- All checks logged before abort

---

## 9. Testing & Validation

### Unit tests
- hash64 deterministic
- merge_sorted_unique correctness
- adjacencies correct
- small hard-coded graphs

### Integration tests
- Multi-rank correctness
- Matching validity after each phase

### Scale tests
- Random graphs
- Power-law graphs
- Monitor ball sizes & stalling

---

## 10. Debugging Playbook

- Ball sizes spike → T_phase too high or sampling too dense
- Rank 0 bottleneck → switch finishing to distributed
- Alltoallv slow → message packing incorrect
- Memory spikes → leakage in ball storage or adjacency rebuild

---

## 11. End of Document
