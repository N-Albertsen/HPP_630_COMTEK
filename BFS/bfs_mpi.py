"""Exercise 1C — Distributed BFS with MPI.   Run: mpiexec -n <P> python bfs_mpi.py"""
import math
from mpi4py import MPI
from bfs_sequential import make_graph


def mpi_bfs(adj, s, comm):
    """Vertex v owned by rank (v % P). Frontier exchanged each level via alltoall."""
    rank, P = comm.Get_rank(), comm.Get_size()
    V = len(adj)
    dist = [math.inf] * V
    front = []
    if s % P == rank:
        dist[s] = 0; front = [s]

    while True:
        out = [[] for _ in range(P)]
        for u in front:
            for w in adj[u]:
                out[w % P].append((w, dist[u] + 1))
        inb = comm.alltoall(out)

        front = []
        for msgs in inb:
            for w, d in msgs:
                if dist[w] == math.inf:
                    dist[w] = d; front.append(w)

        if comm.allreduce(len(front), op=MPI.SUM) == 0:
            return dist


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    for label, n, seed in [("G1", 50_000, 0), ("G2", 500_000, 1)]:
        g = make_graph(n, seed=seed)
        ts = []
        for _ in range(3):
            comm.Barrier()
            t0 = MPI.Wtime(); mpi_bfs(g, 0, comm); ts.append(MPI.Wtime() - t0)
        if comm.Get_rank() == 0:
            print(f"{label}: V={n:,}  P={comm.Get_size()}  avg={sum(ts)/len(ts)*1000:.1f} ms")
