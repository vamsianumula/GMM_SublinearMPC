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

def run_driver_with_io(comm: MPI.Comm, config: MPCConfig, input_path: str):
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    local_edges, local_ids = load_and_distribute_graph(comm, input_path)
    edge_state = init_edge_state(local_edges, local_ids)
    
    total_matches = []
    
    if rank == 0: 
        print(f"[Driver] Loaded. Edges: {len(local_edges)}")
    
    for phase in range(12):
        n_active = np.sum(edge_state.active_mask)
        global_active = comm.allreduce(n_active, op=MPI.SUM)
        
        if rank == 0: 
            print(f"=== Phase {phase} | Active: {global_active} ===")
            
        if global_active == 0: 
            break
            
        # 1. Sparsify
        p_val = 0.5 
        part = sparsify.compute_phase_participation(edge_state, phase, 0, p_val)
        sparsify.compute_deg_in_sparse(comm, edge_state, part, size)
        
        # 2. Stall
        stall.apply_stalling(edge_state, phase, config)
        
        # 3. Exponentiate
        exponentiate.build_balls(comm, edge_state, config, participating_mask=part)
        
        # 4. MIS (Pass the mask!)
        chosen = local_mis.run_greedy_mis(edge_state, phase, participating_mask=part)
        
        # 5. Integrate
        new_m = integrate.update_matching_and_prune(comm, edge_state, chosen, size)
        total_matches.extend(new_m)
        
    extra = finish.finish_small_components(comm, edge_state, config)
    total_matches.extend(extra)
    
    all_lists = comm.gather(total_matches, root=0)
    final_matching = []
    
    if rank == 0:
        for l in all_lists: 
            final_matching.extend(l)
        print(f"Done. Matching Size: {len(final_matching)}")
        
    return final_matching