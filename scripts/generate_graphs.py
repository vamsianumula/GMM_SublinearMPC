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

import argparse

def generate_custom(type, n, p=None, d=None, output="custom.txt"):
    print(f"Generating {type} Graph (n={n})...")
    if type == "dense":
        G = nx.erdos_renyi_graph(n, p, seed=42)
    elif type == "regular":
        G = nx.random_regular_graph(d, n, seed=42)
    elif type == "star":
        G = nx.star_graph(n-1)
    else:
        print(f"Unknown type: {type}")
        return
    write_edge_list(G, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["dense", "regular", "star", "all"], default="all")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--p", type=float, default=0.05)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--out", default="graph.txt")
    args = parser.parse_args()
    
    if args.type == "all":
        random.seed(42)
        generate_dense(n=1000, p=0.05)
        generate_star(n=1000)
        generate_regular(n=1000, d=3)
        generate_dense_small(n=100, p=0.1)
    else:
        generate_custom(args.type, args.n, args.p, args.d, args.out)
