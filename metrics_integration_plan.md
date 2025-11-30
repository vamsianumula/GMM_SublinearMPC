# Metrics Integration Plan for Strongly Sublinear MPC Maximal Matching

This document describes a clean and maintainable strategy for integrating the
full metrics suite specified in the **v2.1 Metrics & Plots Specification**
into the existing codebase.  
The plan covers module changes, data flow, MPI usage, and categories of
metrics, and ends with a practical implementation roadmap.

---

# 0. Objectives

We want the MPC implementation to:

- Track *all* algorithmic, correctness, safety, and system metrics.
- Dump structured data to disk (JSON / CSV) each run.
- Enable offline plotting without modifying core algorithm logic.
- Avoid disrupting the readability of algorithm code.
- Respect MPI correctness and allow rank 0 to aggregate metrics.

Thus we introduce a modular **metrics subsystem** and add lightweight
instrumentation hooks across the codebase.

---

# 1. Architectural Additions

## 1.1 New Module: `metrics.py`

Add a file `src/metrics.py` containing:

### **Phase-level container**

```python
@dataclass
class PhaseMetrics:
    phase_idx: int
    active_edges: int
    matching_size: int
    delta_est: int

    stalling_rate: float
    stalling_rate_by_bucket: Dict[str, float]

    ball_max: int
    ball_mean: float
    ball_p95: float

    mis_selection_rate: float

    # Optional heavy fields:
    ball_growth_factors: List[float]
    comm_volume_bytes: int
    wait_time: float
```

### **Run-level container**

```python
@dataclass
class RunMetrics:
    phases: List[PhaseMetrics]
    total_phases: int
    global_matching_size: int
    unmatched_vertices: int
    max_message_size_bytes: int

    # Correctness checks
    max_incident_matched_edges: int
    edge_consistency_error: int
    symmetric_id_failures: int

    # Test-mode only:
    approximation_ratio: Optional[float]
    maximality_violation_rate: Optional[float]
```

### **Logger**

```python
class MetricsLogger:
    def __init__(self, config, comm):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.run = RunMetrics(...)
        ...

    def log_phase(self, phase_metrics: PhaseMetrics):
        self.run.phases.append(phase_metrics)

    def finalize_and_dump(self, output_dir):
        if self.rank == 0:
            # write run.json and phases.csv
            ...
```

---

# 1.2 Config Upgrades

Extend `MPCConfig`:

- `enable_metrics: bool`
- `enable_test_mode: bool`
- `enable_heavy_metrics: bool`
- `metrics_output_dir: str`

Update `cli.py` to accept:

- `--metrics`
- `--test-mode`
- `--heavy-metrics`
- `--metrics-out PATH`

---

# 2. Instrumentation Plan (Module-by-Module)

We now add metric hooks *without cluttering algorithm logic*.

---

# 2.1 `driver.py` — Central Logging Integration

### Add at top:

```python
logger = MetricsLogger(config, comm) if config.enable_metrics else None
```

### Per-phase:

After each algorithm step, compute metric fragments:

- active edges  
- matching size  
- delta estimate  
- stalling rates  
- deg_in_sparse stats  
- ball_size stats  
- MIS selection rate  
- invariant violations  
- comm volume, wait time  
- max message size  

Construct:

```python
pm = PhaseMetrics(...)
```

Then:

```python
logger.log_phase(pm)
```

### End of run:

- Compute unmatched vertices
- Compute test-mode metrics (approx ratio, maximality)
- `logger.finalize_and_dump(config.metrics_output_dir)`

---

# 2.2 Sparsify (`sparsify.py`)

Add metrics:

### Degree-related:

After `compute_deg_in_sparse`:

- Extract `deg_in_sparse[participating & active]`
- Compute:
  - max, mean, p95, p99
- Send to driver for logging

---

# 2.3 Stalling (`stall.py`)

Before exiting `apply_stalling`:

- Count `new_stalls`
- Bucket edges by degree (precomputed in sparsify)
- Compute:
  - `stalling_rate`
  - `stalling_rate_by_bucket`

Send to driver.

---

# 2.4 Exponentiation (`exponentiate.py`)

During `build_balls`:

- Track per-round total ball volume
- Track ball size max/mean/p95 after final R rounds

Use:

```python
lengths = edge_state.ball_offsets[1:] - edge_state.ball_offsets[:-1]
```

Report:

- `ball_max`
- `ball_mean`
- `ball_p95`
- `ball_growth_factors`

Also register:

- total bytes sent/received (via MPI tracker)

---

# 2.5 MIS (`local_mis.py`)

Metrics:

- Count `candidates` (active & non-stalled & participating)
- Count `chosen`
- MIS selection rate = chosen / candidates

Send to driver.

---

# 2.6 Integration (`integrate.py`)

Metrics after integrating decisions:

### Matching Size

- Each rank counts `len(new_matches_local)`
- Allreduce

### Invariant Checks

- Using vertex owners and `exchange_buffers`, compute:
  - `max_incident_matched_edges_per_vertex`
  - Edge consistency error via sum degree formula
  - Symmetric ID failures (sampled)

Send to driver.

---

# 2.7 Finishing (`finish.py`)

- Count unmatched vertices using final matched vertices.
- In test mode, gather entire graph to rank 0:
  - Compute:
    - approximation ratio
    - maximality violation rate
- Component analysis (if `enable_heavy_metrics`):
  - Gather residual graph
  - Run BFS/DFS
  - Count:
    - number of components
    - max size
    - p95 size

Send results to run metrics.

---

# 2.8 `mpi_helpers.py` — Communication Metrics

Modify `exchange_buffers`:

- Accept a `metrics_tracker` object.
- Track:
  - `bytes_sent` = total_send_elements * dtype.itemsize
  - `bytes_received`
  - max message size in each call
  - time spent inside Alltoall / Alltoallv

Accumulate totals inside `MetricsLogger`.

---

# 3. Metric Computations in Detail

## 3.1 Global Algorithmic Metrics

- Matching size: allreduce count of `matched_edge == True`
- Active edges: count of `active_mask`
- Δ_est: aggregated vertex degrees via MPI
- Unmatched vertices: distributed vertex-matched reduction
- Approx ratio: compute on rank 0 for small graphs
- Total phases/rounds: tracked explicitly

---

## 3.2 Correctness Metrics

- Matching invariant: sum incidents per vertex
- Maximality: edges (u,v) where both endpoints unmatched
- Edge consistency:  
  `| sum(deg)/2 - active_edges |`
- Symmetric ID failures: sampled eid recomputation

---

## 3.3 Phase Diagnostics

- deg_in_sparse stats
- stalling rate + degree buckets
- ball size stats + growth factors
- MIS selection rate

---

## 3.4 Residual Components

- BFS/DFS on gathered final active graph
- Compute CC stats
- Only in test-mode or heavy-metrics mode

---

## 3.5 System Metrics

- Memory usage (RSS)
- Ball storage footprint
- Communication volume per step
- MPI wait times
- Max message size
- Per-rank ball maxima
- Edge/vertex skew

---

# 4. Dump Formats

## 4.1 JSON

`metrics_run.json`:

```json
{
  "total_phases": ...,
  "global_matching_size": ...,
  "unmatched_vertices": ...,
  "max_incident_matched_edges": ...,
  "approximation_ratio": ...,
  "phases": [
    {
      "phase_idx": 0,
      "active_edges": 12345,
      "matching_size": 5500,
      "stalling_rate": 0.13,
      ...
    },
    ...
  ]
}
```

## 4.2 CSV

`metrics_phases.csv`:

```
phase,active_edges,matching_size,ball_max,ball_mean,mis_rate,stall_rate,...
0,12000,500,...
1,8000,580,...
```

---

# 5. Plotting

Create `tools/plot_metrics.py`:

- Reads JSON/CSV
- Generates:
  - active edges decay
  - matching growth
  - delta_est decay
  - ball size distributions
  - stalling rate
  - memory per rank
  - comm volume breakdown
  - CC size histogram

---

# 6. Implementation Roadmap

## Step 1: Add `metrics.py` module  
Implement containers + JSON/CSV dump logic + tracker API.

## Step 2: Modify `mpi_helpers.exchange_buffers`  
Add counters & timers to feed into the tracker.

## Step 3: Add metric hooks in `driver.py`  
Compute per-phase summaries; gather values; call logger.

## Step 4: Add metric code to sparsify/stall/exponentiate/MIS/integrate  

## Step 5: Add test-mode metrics to `finish.py`  

## Step 6: Add CLI flags & `MPCConfig` fields  

## Step 7: Write plotting tools (not part of algorithm run)  

---

# 7. Conclusion

This plan integrates all metrics in a maintainable, modular, and MPI-correct way.
The algorithm code remains clean, while rank 0 receives accurate summaries for
plotting and analysis. All work-intensive or large-data plotting is performed
offline.
