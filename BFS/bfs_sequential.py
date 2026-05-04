"""Exercise 1A — Sequential BFS"""
import time, math, random
from collections import deque


def make_graph(n, m=5, seed=0):
    """Barabási–Albert scale-free graph as adjacency list."""
    rng = random.Random(seed)
    adj = {v: [] for v in range(n)}
    stubs = []
    for u in range(m + 1):
        for v in range(u + 1, m + 1):
            adj[u].append(v); adj[v].append(u)
            stubs += [u, v]
    for v in range(m + 1, n):
        targets = set()
        while len(targets) < m:
            targets.add(rng.choice(stubs))
        for t in targets:
            adj[v].append(t); adj[t].append(v)
            stubs += [v, t]
    return adj


def bfs(adj, s):
    """Queue-based BFS. Work = Θ(V+E)."""
    dist = {v: math.inf for v in adj}
    dist[s] = 0
    Q = deque([s])
    while Q:
        u = Q.popleft()
        for w in adj[u]:
            if dist[w] == math.inf:
                dist[w] = dist[u] + 1
                Q.append(w)
    return dist


def avg_time(fn, *args, runs=5):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); fn(*args); ts.append(time.perf_counter() - t0)
    return sum(ts) / len(ts)


if __name__ == "__main__":
    for label, n, seed in [("G1", 50_000, 0), ("G2", 500_000, 1)]:
        g = make_graph(n, seed=seed)
        e = sum(map(len, g.values())) // 2
        t = avg_time(bfs, g, 0)
        print(f"{label}: V={n:,}  E={e:,}  avg={t*1000:.1f} ms")
