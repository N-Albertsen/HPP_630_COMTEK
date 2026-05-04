"""
Main runner — executes every BFS implementation and prints a summary.

Usage:  python main.py
        (MPI ranks are launched via subprocess if mpiexec is on PATH)
"""
import shutil
import subprocess
import time

from numba import set_num_threads

from bfs_sequential import make_graph, bfs, avg_time
from bfs_parallel    import (parallel_bfs_threads, parallel_bfs_numba,
                             to_csr, MAX_NUMBA)
from bfs_demo        import bfs_levels, adj as demo_adj


GRAPHS  = [("G1",  50_000, 0),
           ("G2", 500_000, 1)]
P_VALUES = (1, 2, 4, 8)
RUNS    = 5


# ──────────────────────────────────────────────────────────────────
# B.II — Small illustrative graph
# ──────────────────────────────────────────────────────────────────
def run_demo():
    print("─" * 60)
    print("B.II — Level-by-level BFS DAG on a 7-node graph")
    print("─" * 60)
    for k, F in enumerate(bfs_levels(demo_adj, 0)):
        tasks = "   ".join(f"v{v}" for v in F)
        print(f"  L{k}:  {tasks}     ← parallel within level")


# ──────────────────────────────────────────────────────────────────
# In-process benchmarks (sequential + threads + numba)
# ──────────────────────────────────────────────────────────────────
def benchmark_inprocess(label, n, seed):
    g = make_graph(n, seed=seed)
    e = sum(map(len, g.values())) // 2
    offs, nbrs = to_csr(g)
    V = len(g)
    parallel_bfs_numba(offs, nbrs, V, 0)         # JIT warm-up

    results = {"V": n, "E": e}
    results[("Sequential", "-")] = avg_time(bfs, g, 0, runs=RUNS)

    for P in P_VALUES:
        results[("Threads", P)] = avg_time(parallel_bfs_threads, g, 0, P, runs=RUNS)
        set_num_threads(min(P, MAX_NUMBA))
        results[("Numba",   P)] = avg_time(parallel_bfs_numba, offs, nbrs, V, 0, runs=RUNS)

    print(f"\n{label}:  V = {n:,}   E = {e:,}")
    return results


# ──────────────────────────────────────────────────────────────────
# MPI via subprocess (if mpiexec is available)
# ──────────────────────────────────────────────────────────────────
def benchmark_mpi():
    """Returns {(graph_label, P): seconds} or {} if mpiexec missing."""
    if not shutil.which("mpiexec"):
        print("\n[MPI skipped — mpiexec not on PATH]")
        return {}
    out = {}
    for P in P_VALUES:
        try:
            r = subprocess.run(
                ["mpiexec", "-n", str(P), "python", "bfs_mpi.py"],
                capture_output=True, text=True, timeout=600, check=True,
            )
            for line in r.stdout.splitlines():
                if ":" in line and "avg=" in line:
                    label = line.split(":", 1)[0].strip()
                    ms = float(line.rsplit("avg=", 1)[1].split()[0])
                    out[(label, P)] = ms / 1000
        except Exception as exc:
            print(f"  MPI P={P} failed: {exc}")
    return out


# ──────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────
def print_summary(in_proc, mpi_results):
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)

    for label, _, _ in GRAPHS:
        r = in_proc[label]
        t_seq = r[("Sequential", "-")]
        rows = [("Sequential", "-", t_seq)]
        for P in P_VALUES:
            rows.append(("Threads", P, r[("Threads", P)]))
            rows.append(("Numba",   P, r[("Numba",   P)]))
            if (label, P) in mpi_results:
                rows.append(("MPI", P, mpi_results[(label, P)]))

        best = min(rows[1:], key=lambda x: x[2])
        print(f"\n{label}:  V = {r['V']:,}   E = {r['E']:,}")
        print(f"  {'Method':<12}{'P':>4}  {'Time (ms)':>11}  {'Speedup':>9}")
        for m, P, t in rows:
            tag = "  ← best" if (m, P, t) == best else ""
            print(f"  {m:<12}{str(P):>4}  {t*1000:>11.1f}  {t_seq/t:>8.2f}x{tag}")


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.perf_counter()
    print("═" * 60)
    print("BFS WORKSHOP — Combined Benchmark Runner")
    print("═" * 60)

    run_demo()

    in_proc = {}
    for label, n, seed in GRAPHS:
        in_proc[label] = benchmark_inprocess(label, n, seed)

    mpi_results = benchmark_mpi()

    print_summary(in_proc, mpi_results)
    print(f"\nTotal wall-clock: {time.perf_counter() - t0:.1f} s")
