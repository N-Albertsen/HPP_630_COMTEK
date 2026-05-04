from src.graph import generate_connected_graph, to_csr
from src.bfs_sequential import bfs
from src.bfs_parallel import parallel_bfs
from src.bfs_multiprocessing import multiprocessing_bfs
from src.visualize import plot_graph
from src.bfs_numba import bfs_numba
import numpy as np
import time


def measure_time(func, graph, start, runs=3, **kwargs):
    times = []
    for _ in range(runs):
        t0 = time.time()
        func(graph, start, **kwargs)
        t1 = time.time()
        times.append(t1 - t0)
    return sum(times) / len(times)


def main():
    # ONE graph (used for both)
    graph = generate_connected_graph(10000, 30000)
    print(f"Generated graph with {len(graph)} nodes and {sum(len(neighbors) for neighbors in graph.values()) // 2} edges.")

    # Sequential BFS
    t_seq = measure_time(bfs, graph, 0)
    print(f"Sequential BFS time: {t_seq:.6f} seconds")

    # Parallel BFS (setting num_threads)
    threads = [1, 2, 4, 8]

    print("\nParallel BFS times:")
    for t in threads:
        t_par = measure_time(parallel_bfs, graph, 0, num_threads=t)
        speedup = t_seq / t_par
        print(f"Threads: {t} | Time: {t_par:.6f} | Speedup vs seq: {speedup:.2f}")

    print("\nMultiprocessing BFS times:")
    for p in threads:
        t_mp = measure_time(multiprocessing_bfs, graph, 0, num_processes=p)
        speedup = t_seq / t_mp
        print(f"Processes: {p} | Time: {t_mp:.6f} | Speedup vs seq: {speedup:.2f}")

    offsets, neighbors = to_csr(graph)

    # warm-up (IMPORTANT for Numba)
    bfs_numba(offsets, neighbors, 0)

    t_numba = measure_time(
        lambda g, s: bfs_numba(offsets, neighbors, s),
        graph,
        0
    )

    print(f"\nNumba BFS time: {t_numba:.6f}")
    print(f"Speedup vs seq: {t_seq / t_numba:.2f}")

    # Plot (skipped automatically if graph too large)
    print("\nShowing graph visualization")
    plot_graph(graph)

if __name__ == "__main__":
    main()