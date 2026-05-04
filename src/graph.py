import random
import numpy as np

def generate_connected_graph(num_nodes, num_edges):
    graph = {i: [] for i in range(num_nodes)}

    # Step 1: Create a connected structure (tree)
    for i in range(1, num_nodes):
        j = random.randint(0, i - 1)
        graph[i].append(j)
        graph[j].append(i)

    # Step 2: Add extra edges
    extra_edges = num_edges - (num_nodes - 1)

    for _ in range(extra_edges):
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)

        if u != v:
            graph[u].append(v)
            graph[v].append(u)

    return graph

def to_csr(graph):
    n = len(graph)
    offsets = [0]
    neighbors = []

    for i in range(n):
        nbrs = graph[i]
        neighbors.extend(nbrs)
        offsets.append(len(neighbors))

    return np.array(offsets), np.array(neighbors)