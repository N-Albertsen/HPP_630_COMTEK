"""Exercise 1B — Level-synchronous parallel BFS using BOTH threads and Numba"""
import time, math, threading
import numpy as np
from numba import njit, prange, set_num_threads, config
from bfs_sequential import make_graph, bfs, avg_time

MAX_NUMBA = config.NUMBA_NUM_THREADS   # max threads numba can launch on this machine


# ── Threads version (Python + lock) ─────────────────────────────────
def parallel_bfs_threads(adj, s, P):
    V = len(adj)
    dist = [math.inf] * V
    dist[s] = 0
    frontier = [s]
    nxt = []
    lock = threading.Lock()
    bar = threading.Barrier(P)

    def worker(tid):
        nonlocal frontier, nxt
        while True:
            local = []
            chunk = max(1, math.ceil(len(frontier) / P))
            for u in frontier[tid * chunk : (tid + 1) * chunk]:
                for w in adj[u]:
                    if dist[w] == math.inf:
                        with lock:
                            if dist[w] == math.inf:
                                dist[w] = dist[u] + 1
                                local.append(w)
            with lock: nxt.extend(local)
            bar.wait()
            if tid == 0: frontier, nxt = nxt, []
            bar.wait()
            if not frontier: return

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(P)]
    for t in ts: t.start()
    for t in ts: t.join()
    return dist


# ── Numba version (CSR + prange, race-tolerant) ─────────────────────
def to_csr(adj):
    """Convert dict adjacency list to CSR arrays for Numba."""
    V = len(adj)
    offsets = np.zeros(V + 1, dtype=np.int64)
    for v in range(V):
        offsets[v + 1] = offsets[v] + len(adj[v])
    neighbors = np.empty(offsets[-1], dtype=np.int64)
    for v in range(V):
        for i, w in enumerate(adj[v]):
            neighbors[offsets[v] + i] = w
    return offsets, neighbors


@njit(parallel=True, cache=True)
def _explore(offsets, neighbors, dist, frontier, n_front, level):
    """Parallel frontier expansion. Race on dist[w] is benign:
    every writer assigns the same value (level+1)."""
    for i in prange(n_front):
        u = frontier[i]
        for j in range(offsets[u], offsets[u + 1]):
            w = neighbors[j]
            if dist[w] == -1:
                dist[w] = level + 1


@njit(cache=True)
def _collect(dist, V, target_level, out):
    """Sequential O(V) scan to gather the next frontier."""
    n = 0
    for v in range(V):
        if dist[v] == target_level:
            out[n] = v; n += 1
    return n


def parallel_bfs_numba(offsets, neighbors, V, s):
    dist = np.full(V, -1, dtype=np.int64)
    dist[s] = 0
    frontier = np.empty(V, dtype=np.int64)
    frontier[0] = s
    n_front, level = 1, 0
    while n_front > 0:
        _explore(offsets, neighbors, dist, frontier, n_front, level)
        n_front = _collect(dist, V, level + 1, frontier)
        level += 1
    return dist


# ── Main: compare sequential, threads, numba ────────────────────────
if __name__ == "__main__":
    for label, n, seed in [("G1", 50_000, 0), ("G2", 500_000, 1)]:
        g = make_graph(n, seed=seed)
        offsets, neighbors = to_csr(g)
        V = len(g)

        parallel_bfs_numba(offsets, neighbors, V, 0)   # JIT warm-up

        t_seq = avg_time(bfs, g, 0)
        print(f"\n{label}: V={n:,}   sequential = {t_seq*1000:.1f} ms")
        print(f"  {'method':<10}{'P':>3}  {'time (ms)':>10}  {'speedup':>9}  {'eff':>5}")

        for P in (1, 2, 4, 8):
            t = avg_time(parallel_bfs_threads, g, 0, P)
            print(f"  {'threads':<10}{P:>3}  {t*1000:>10.1f}  {t_seq/t:>8.2f}x  {t_seq/t/P:>4.0%}")

            set_num_threads(min(P, MAX_NUMBA))
            t = avg_time(parallel_bfs_numba, offsets, neighbors, V, 0)
            print(f"  {'numba':<10}{P:>3}  {t*1000:>10.1f}  {t_seq/t:>8.2f}x  {t_seq/t/P:>4.0%}")
