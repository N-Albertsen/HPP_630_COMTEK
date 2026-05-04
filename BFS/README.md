# Workshop 1 — BFS

## Requirements

```bash
pip install mpi4py    # only for bfs_mpi.py
```

Plus an MPI runtime (OpenMPI or MPICH) on `PATH`.

## Run

```bash
python bfs_sequential.py        # 1A
python bfs_parallel.py          # 1B
mpiexec -n 4 python bfs_mpi.py  # 1C — change -n for more/fewer processes
```

Keep all three files in the same directory (`bfs_parallel.py` and `bfs_mpi.py` import from `bfs_sequential.py`).
