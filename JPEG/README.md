# JPEG DCT Compression — COMTEK 6 / ESD 6
## Workshop Assignment — May 2026

---

## Files

| File | Description |
|------|-------------|
| `jpeg_dct.py` | Core pipeline: Steps 1–5, all 4 implementations |
| `run_benchmark.py` | **Main script** — runs Tasks IV, V, VI, VII |
| `jpeg_mpi.py` | MPI distributed implementation (Task VII) |
| `complexity_dag.py` | Task I (complexity) + Task II (DAG plots) |
| `requirements.txt` | Python dependencies |

---

## Installation

```bash
pip install -r requirements.txt
```

For MPI support (optional, needed for `jpeg_mpi.py`):
```bash
# Ubuntu/Debian
sudo apt install openmpi-bin libopenmpi-dev
pip install mpi4py

# macOS
brew install open-mpi
pip install mpi4py
```

---

## Usage

### Run all benchmarks (Tasks IV–VII)
```bash
python run_benchmark.py                     # uses synthetic test image
python run_benchmark.py my_photo.jpg        # uses your image
```

### Generate complexity analysis + DAG plots (Tasks I–II)
```bash
python complexity_dag.py
```

### Run with MPI (Task VII, requires cluster or local MPI)
```bash
mpiexec -n 4 python jpeg_mpi.py my_photo.jpg 8 8
mpiexec -n 8 python jpeg_mpi.py my_photo.jpg 8 8
```

---

## Task I: Work and Complexity

| Step | Operation | Work | Depth |
|------|-----------|------|-------|
| 1 | Grayscale conversion + centering | O(N·M) | O(1) |
| 2 | Block partitioning (view) | O(1)–O(N·M) | O(1) |
| 3 | 2D DCT on all blocks | O(N·M·(Kh+Kw)) | O(Kh·Kw·(Kh+Kw)) |
| 4 | Quantization | O(N·M) | O(1) |
| 5 | 2D IDCT (reconstruction) | O(N·M·(Kh+Kw)) | O(Kh·Kw·(Kh+Kw)) |

**Total**: O(N·M·(Kh+Kw))

- For **Kh=Kw=8**: O(16·N·M) — linear, very efficient
- For **Kh=N, Kw=M**: O(N·M·(N+M)) — cubic for square images

The DCT step dominates. Steps 3 and 5 have the same complexity since
the IDCT matrix is simply D^T / (2N), requiring identical operations.

### Why the block approach is efficient (Kh=Kw=8)
Each 8×8 DCT costs O(8·8·16) = 1024 ops. There are (N/8)·(M/8) = N·M/64
blocks. Total: (N·M/64) · 1024 = 16·N·M operations. The constant factor 16
makes 8×8 JPEG far cheaper than a single N×M DCT.

---

## Task II: DAG Structure

Run `python complexity_dag.py` to generate the DAG plots.

**Key observations:**
- Steps 1 and 4 are embarrassingly parallel (per-pixel independence)
- Step 2 is a data reorganisation — O(1) with NumPy views
- Steps 3 and 5: all *blocks* are independent → parallelise over blocks
- Within one block, *rows* of the DCT can run in parallel, then *columns*
- The only synchronisation barrier is between row-DCTs and column-DCTs within a block

---

## Task III: Method Ranking (Discussion)

### For Kh=Kw=8 (small blocks, many of them)

**Sequential:**
1. **NumPy vectorized** — batch all blocks as a 4D tensor, use matmul broadcasting.
   Very fast due to BLAS. No Python overhead per block.
2. **Numba JIT** — fast after compilation, but less cache-friendly than NumPy BLAS.

**Parallel:**
1. **Numba JIT (parallel=True)** — prange over blocks, low overhead, shared memory.
2. **Multiprocessing** — high serialisation/IPC cost for small blocks (pickle overhead
   per chunk dominates). Better for large blocks.
3. **Threading** — limited by Python GIL for pure Python; Numba releases GIL.
4. **MPI** — communication overhead too high for local runs; shines on clusters.

### For Kh=N, Kw=M (one large block)

**Sequential:**
1. **NumPy vectorized** — single large matrix multiply, BLAS-optimised.
2. **Numba JIT** — slower than NumPy BLAS for large matrices.

**Parallel:**
1. **Multiprocessing** — only 1 block, so parallelism must be *within* the DCT itself.
   Useful if splitting rows/columns across processes.
2. **Numba parallel** — can parallelise inner loops; useful here.
3. **MPI** — row/column distribution across nodes is feasible.

---

## Task IV: Implemented Methods

| Method | Type | Key technique |
|--------|------|---------------|
| `JPEGNumpy` | Sequential | 4D tensor matmul via `np.tensordot` |
| `JPEGNumbaSeq` | Sequential | `@njit` compiled loops |
| `JPEGNumbaParallel` | Parallel | `@njit(parallel=True)` + `prange` |
| `JPEGMultiprocessing` | Parallel | `mp.Pool` scatter/gather over chunks |

---

## Task V: Execution Time

Run `run_benchmark.py` to generate:
- `output/timing_block8.png` — timing for Kh=Kw=8 on two image sizes
- `output/timing_full_block.png` — timing for full-image blocks
- `output/worker_sweep.png` — optimal worker count for multiprocessing

**Numba note**: First call includes JIT compilation time (~1–5s).
Subsequent calls (n_runs=5) reflect true runtime.

---

## Task VI: Quality Evaluation

Compression factor multiplies the quantization matrix Q.

| Factor | Expected PSNR | Quality Score |
|--------|---------------|---------------|
| 0.1    | >45 dB        | 5/5 (near lossless) |
| 0.5    | ~42 dB        | 5/5 (excellent) |
| 1.0    | ~38 dB        | 4/5 (standard JPEG) |
| 2.0    | ~34 dB        | 3–4/5 (good) |
| 4.0    | ~28 dB        | 2–3/5 (visible artefacts) |
| 8.0    | ~24 dB        | 1–2/5 (blocky) |
| 16.0   | ~20 dB        | 0–1/5 (horrible) |

Run `python run_benchmark.py` to see actual values for your image.

---

## Task VII: MPI Speedup Analysis

**Communication model** (R = 200 Mbps):
- Transmit 1 float32 (32-bit): t₃₂ = 32 / (200×10⁶) = 160 ns
- Transmit 1 int8 (8-bit):     t₈  =  8 / (200×10⁶) =  40 ns

**Cost breakdown for P processes:**
- Scatter image data (N·M floats): T_scatter = (N·M · 32) / R
- Per-process compute:             T_compute = T_seq / P
- Gather results (N·M floats):    T_gather  = (N·M · 32) / R
- **Total**: T(P) = 2·(N·M·32)/R + T_seq/P

**Maximum speedup** (Amdahl's law applied to communication):
```
S_max = T_seq / (2 · N·M · 32 / R)
```

For a 512×512 image and T_seq ≈ 0.1s:
- T_comm = 2 · (512·512·32) / (200×10⁶) ≈ 0.042 s
- S_max ≈ 0.1 / 0.042 ≈ **2.4×**

For a 256×256 image and T_seq ≈ 0.02s:
- T_comm ≈ 0.011 s
- S_max ≈ 0.02 / 0.011 ≈ **1.8×**

**Conclusion**: With R=200 Mbps, the communication cost is a significant
fraction of compute time for typical image sizes. MPI is only beneficial
for very large images or when T_seq >> T_comm. On a faster network
(e.g., InfiniBand at 100 Gbps), S_max would scale proportionally.

The optimal P is where the marginal speedup gain equals the marginal
communication increase. In practice, P = 4–8 is a good trade-off.

---

## Notes for Report

- All methods implement the exact same JPEG pipeline (Steps 1–5)
- No scipy.fftpack used — DCT is computed from definition / matrix method
- Quantization matrices: standard JPEG 8×8 for Kh=Kw=8; generated for other sizes
- For large block sizes, quantization matrix is frequency-scaled (see `make_quantization_matrix`)
- Numba functions are cached (`cache=True`) so second run is faster
