"""
src/mm_mpc/driver.py
Main Driver.
"""
import time
import numpy as np
from mpi4py import MPI
from .config import MPCConfig
from .graph_io import load_and_distribute_graph
from .state_layout import init_edge_state
from .phases import sparsify, stall, exponentiate, local_mis, integrate, finish

from .metrics import MetricsLogger, PhaseMetrics

def run_driver_with_io(comm: MPI.Comm, config: MPCConfig, input_path: str):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Metrics Init
    logger = MetricsLogger(config, comm) if config.enable_metrics else None
    
    local_edges, local_ids = load_and_distribute_graph(comm, input_path)
    edge_state = init_edge_state(local_edges, local_ids)
    
    total_matches = []
    
    if rank == 0: 
        print(f"[Driver] Loaded. Edges: {len(local_edges)}")
    
    if rank == 0: 
        print(f"[Driver] Loaded. Edges: {len(local_edges)}")
    
    # Adaptive State
    prev_max_ball = 1
    
    # Increased max phases to allow for slower convergence with adaptive throttling
    for phase in range(30):
        if logger: logger.tracker.reset_phase()
        
        n_active = np.sum(edge_state.active_mask)
        global_active = comm.allreduce(n_active, op=MPI.SUM)
        
        if rank == 0: 
            print(f"=== Phase {phase} | Active: {global_active} ===")
            
        if global_active == 0: 
            break
            
        # 1. Adaptive Sparsification Probability
        p_val = 0.5
        if config.adaptive_sparsification:
            est_ball_size = max(1, prev_max_ball * 2)
            # Total Capacity = P * S * Safety
            total_capacity = size * config.S_edges * config.safety_factor
            # Estimated Load = Active * EstBallSize
            est_load = global_active * est_ball_size
            
            if est_load > total_capacity:
                p_val = total_capacity / est_load
                # Clamp to reasonable bounds
                p_val = max(0.0001, min(0.5, p_val))
                
            if rank == 0:
                print(f"    [Adaptive] EstBall: {est_ball_size}, Load: {est_load:.0f}, Cap: {total_capacity:.0f} -> p={p_val:.6f}")

        # 2. Sparsify
        part = sparsify.compute_phase_participation(edge_state, phase, 0, p_val)
        spars_stats = sparsify.compute_deg_in_sparse(comm, edge_state, part, size)
        
        # 3. Stall
        stall_stats = stall.apply_stalling(edge_state, phase, config)
        
        # 4. Exponentiate
        # Pass tracker if metrics enabled
        tracker = logger.tracker if logger else None
        ball_stats = exponentiate.build_balls(comm, edge_state, config, participating_mask=part, tracker=tracker)
        
        # 5. MIS
        # TODO: Update local_mis to accept tracker if we want to track its comm too
        chosen, mis_rate = local_mis.run_greedy_mis(edge_state, phase, participating_mask=part)
        
        # 6. Integrate
        # TODO: Update integrate to accept tracker
        new_m = integrate.update_matching_and_prune(comm, edge_state, chosen, size)
        total_matches.extend(new_m)
        
        # Metrics Logging
        if logger:
            local_m_size = len(new_m)
            global_m_size = comm.allreduce(local_m_size, op=MPI.SUM)
            
            # Global Reductions for Metrics
            g_ball_max = comm.reduce(ball_stats["max"], op=MPI.MAX, root=0)
            g_ball_mean_sum = comm.reduce(ball_stats["mean"], op=MPI.SUM, root=0)
            
            # Update Adaptive State (Peak Hold Estimator)
            current_max_ball = g_ball_max if rank == 0 else 1
            current_max_ball = comm.bcast(current_max_ball, root=0)
            prev_max_ball = max(prev_max_ball, current_max_ball)
            
            # Max Communication (Rank Bottleneck)
            local_comm_bytes = logger.tracker.bytes_sent
            local_comm_items = logger.tracker.items_sent
            
            g_comm_bytes_max = comm.reduce(local_comm_bytes, op=MPI.MAX, root=0)
            g_comm_items_max = comm.reduce(local_comm_items, op=MPI.MAX, root=0)
            
            if rank == 0:
                g_ball_mean = g_ball_mean_sum / size
                
                pm = PhaseMetrics(
                    phase_idx=phase,
                    active_edges=global_active,
                    matching_size=global_m_size,
                    delta_est=0, # TODO: Implement delta est
                    
                    sparsification_p=p_val,
                    
                    deg_min=spars_stats["min"],
                    deg_max=spars_stats["max"],
                    deg_mean=spars_stats["mean"],
                    deg_p95=spars_stats["p95"],
                    
                    stalling_rate=stall_stats["rate"],
                    stalling_rate_by_bucket={},
                    ball_max=g_ball_max,
                    ball_mean=g_ball_mean,
                    ball_p95=ball_stats["p95"], # Local p95 of rank 0 is proxy
                    mis_selection_rate=mis_rate,
                    comm_volume_bytes=g_comm_bytes_max, # Global Max
                    comm_volume_items=g_comm_items_max, # Global Max
                    wait_time_seconds=logger.tracker.comm_time
                )
                logger.log_phase(pm)

    extra = finish.finish_small_components(comm, edge_state, config)
    total_matches.extend(extra)
    
    all_lists = comm.gather(total_matches, root=0)
    final_matching = []
    
    if rank == 0:
        for l in all_lists: 
            final_matching.extend(l)
        print(f"Done. Matching Size: {len(final_matching)}")
        
        if logger:
            logger.run_metrics.global_matching_size = len(final_matching)
            logger.run_metrics.S_edges = config.S_edges
            logger.run_metrics.n_global = config.n_global
            logger.finalize_and_dump()
        
    return final_matching