#!/usr/bin/env python3
"""
scripts/generate_graphs.py
Generates 3 types of graphs for MPC testing:
1. Dense (Erdos-Renyi)
2. Star (Hub-and-Spoke)
3. Regular (3-Regular)
"""

import networkx as nx
import random
import os

def write_edge_list(G, filename):
    with open(filename, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")
    print(f"Generated {filename}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

def generate_dense(n=1000, p=0.05):
    print(f"Generating Dense Graph (n={n}, p={p})...")
    G = nx.erdos_renyi_graph(n, p, seed=42)
    write_edge_list(G, "dense.txt")

def generate_star(n=1000):
    print(f"Generating Star Graph (n={n})...")
    G = nx.star_graph(n-1) # n nodes total
    write_edge_list(G, "star.txt")

def generate_regular(n=1000, d=3):
    print(f"Generating {d}-Regular Graph (n={n})...")
    # random_regular_graph requires n * d to be even
    G = nx.random_regular_graph(d, n, seed=42)
    write_edge_list(G, "regular.txt")

def generate_dense_small(n=100, p=0.1):
    print(f"Generating Small Dense Graph (n={n}, p={p})...")
    G = nx.erdos_renyi_graph(n, p, seed=42)
    write_edge_list(G, "dense_small.txt")

if __name__ == "__main__":
    random.seed(42)
    generate_dense(n=1000, p=0.05) # ~25k edges
    generate_star(n=1000)          # ~1k edges
    generate_regular(n=1000, d=3)  # ~1.5k edges
    generate_dense_small(n=100, p=0.1)  # ~1.5k edges
