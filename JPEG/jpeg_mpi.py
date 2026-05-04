"""
JPEG DCT - MPI Implementation (Task VII)
COMTEK 6 / ESD 6

Run with:  mpiexec -n <P> python jpeg_mpi.py <image_path> [Kh] [Kw]

Requires: mpi4py  (pip install mpi4py)

Communication model (Task VII):
  - Data rate R = 200 Mbps
  - Transmitting w-bit value takes t_w = w/R seconds
  - 32-bit float: t_32 = 32 / (200e6) = 160 ns
  - Scatter/Gather cost dominates at small block counts

Maximum Speedup Analysis (see mpi_speedup_analysis below):
  - The image blocks are independent (embarrassingly parallel for steps 3-5)
  - Scatter cost: O(N*M * 32 / R) to distribute all pixel data
  - Gather cost: O(N*M * 32 / R) to collect results
  - Compute time per process: T_seq / P
  - Speedup: S(P) = T_seq / (T_comm + T_seq/P)
"""

import numpy as np
import sys
import time

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("mpi4py not available. Run: pip install mpi4py")

from jpeg_dct import (
    load_and_convert, pad_image, extract_blocks, reconstruct_from_blocks,
    dct_matrix, idct_matrix, make_quantization_matrix
)

# Communication parameters (Task VII)
R_BPS = 200e6        # 200 Mbps data rate
BITS_FLOAT32 = 32
BITS_INT8 = 8
T_FLOAT32 = BITS_FLOAT32 / R_BPS   # time to send one float32
T_INT8 = BITS_INT8 / R_BPS         # time to send one int8


def mpi_speedup_analysis(N: int, M: int, Kh: int, Kw: int,
                          T_seq: float, P_max: int = 64):
    """
    Task VII: Theoretical maximum speedup analysis for MPI.

    Communication model (wired links, R = 200 Mbps):
      t_w = w / R  seconds to transmit one w-bit value.
      t_32 = 32 / (200e6) = 160 ns per float32.

    MPI strategy: root distributes blocks, each rank computes DCT+Q+IDCT,
    root gathers results.

    Scatter cost (root -> P ranks, switched cluster):
      On a switched cluster each link is independent, so the root can
      send to all P ranks simultaneously. Each rank receives (N*M/P)
      float32 values, so the scatter time is:
        T_scatter(P) = (N*M / P) * t_32

    Gather cost (P ranks -> root): symmetric:
        T_gather(P) = (N*M / P) * t_32

    Compute per rank (blocks are fully independent):
        T_compute(P) = T_seq / P

    Total time:
        T(P) = T_scatter + T_compute + T_gather
             = 2*(N*M/P)*t_32 + T_seq/P
             = (1/P) * (2*N*M*t_32 + T_seq)

    Speedup:
        S(P) = T_seq / T(P)
             = P * T_seq / (T_seq + 2*N*M*t_32)

    Since both compute AND communication scale as 1/P, speedup is linear
    in P. Efficiency is constant (independent of P):
        E = S(P)/P = T_seq / (T_seq + 2*N*M*t_32)

    The overhead fraction alpha = 2*N*M*t_32 / T_seq determines efficiency:
      - alpha << 1: E ≈ 1, near-perfect scaling
      - alpha >> 1: E << 1, communication dominates — MPI not worthwhile

    Upper bound on useful P: limited by number of blocks B = (N/Kh)*(M/Kw).
    Beyond P=B, some ranks get no blocks and speedup saturates.
    """
    t32 = BITS_FLOAT32 / R_BPS          # 160 ns per float32
    T_overhead_total = 2.0 * N * M * t32  # total comm data if sent on one link
    alpha = T_overhead_total / T_seq      # comms-to-compute ratio at P=1

    print("\n" + "="*60)
    print("MPI SPEEDUP ANALYSIS (Task VII)")
    print("="*60)
    print(f"Image size:       {N} x {M} = {N*M} pixels")
    print(f"Block size:       {Kh} x {Kw}")
    print(f"Data rate:        R = {R_BPS/1e6:.0f} Mbps")
    print(f"t_32:             {t32*1e9:.1f} ns per float32")
    print(f"Sequential time:  T_seq = {T_seq:.4f} s")
    print(f"Comm overhead:    2*N*M*t_32 = {T_overhead_total*1e3:.2f} ms")
    print(f"  alpha = T_comm_total/T_seq = {alpha:.3f}")
    print()
    print("Network: switched cluster — scatter/gather each take (N*M/P)*t_32")
    print("=> T(P) = (T_seq + 2*N*M*t_32) / P   [both compute & comms scale as 1/P]")
    print("=> S(P) = P * T_seq / (T_seq + 2*N*M*t_32)  [linear speedup]")
    print()

    E_const = T_seq / (T_seq + T_overhead_total)
    print(f"Parallel efficiency E = {E_const*100:.1f}%  (constant for all P)")
    print()
    print(f"{'P':>6} | {'T(P) [ms]':>12} | {'Speedup':>10} | {'Efficiency':>12}")
    print("-" * 50)

    results = []
    n_blocks = (N // Kh) * (M // Kw)
    for P in sorted(set([1, 2, 4, 8, 16, 32, 64, min(n_blocks, P_max)])):
        T_p = (T_seq + T_overhead_total) / P
        S = T_seq / T_p
        eff = S / P * 100
        results.append((P, T_p, S, eff))
        marker = " ← max useful P" if P == n_blocks else ""
        print(f"{P:>6} | {T_p*1e3:>12.3f} | {S:>10.2f}x | {eff:>10.1f}%{marker}")

    print()
    print(f"Max useful P = number of blocks = {n_blocks}")
    print(f"Max achievable speedup (P={n_blocks}):  {n_blocks * E_const:.1f}x")
    print()
    print("CONCLUSION:")
    if alpha < 0.1:
        verdict = (f"Communication overhead is negligible (alpha={alpha:.3f}). "
                   f"MPI scales almost perfectly — E={E_const*100:.0f}% efficiency.")
    elif alpha < 1.0:
        verdict = (f"Communication overhead is moderate (alpha={alpha:.2f}). "
                   f"MPI gives {E_const*100:.0f}% efficiency with meaningful speedup.")
    else:
        verdict = (f"Communication overhead dominates (alpha={alpha:.1f}). "
                   f"E={E_const*100:.0f}% per process. For a 256x256 image at "
                   f"R=200 Mbps the 21 ms transmission cost exceeds the {T_seq*1e3:.0f} ms "
                   f"compute time. MPI is only beneficial here on much larger images "
                   f"or with a faster interconnect (e.g. InfiniBand ~100 Gbps would "
                   f"give alpha≈{alpha * 200/100000:.4f}, near-perfect efficiency).")
    print(f"  {verdict}")

    return results


def run_mpi_jpeg(image_path: str, Kh: int = 8, Kw: int = 8):
    """
    Distributed JPEG compression using MPI.
    
    Strategy:
      - Rank 0 loads image, pads, extracts blocks
      - Scatter blocks evenly across all ranks
      - Each rank performs DCT + quantize + IDCT on its chunk
      - Gather results back to rank 0
      - Rank 0 reconstructs full image
    """
    if not MPI_AVAILABLE:
        print("mpi4py required. Install with: pip install mpi4py")
        return

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    P = comm.Get_size()

    t_start = MPI.Wtime()

    # --- Rank 0: load and prepare data ---
    if rank == 0:
        img = load_and_convert(image_path)
        H_orig, W_orig = img.shape
        img_pad = pad_image(img, Kh, Kw)
        blocks = extract_blocks(img_pad, Kh, Kw)
        n_bh, n_bw, _, _ = blocks.shape
        blocks_flat = blocks.reshape(-1, Kh, Kw).copy()
        n_blocks = len(blocks_flat)
        Q = make_quantization_matrix(Kh, Kw)
        meta = (H_orig, W_orig, n_bh, n_bw, n_blocks, Kh, Kw)
    else:
        blocks_flat = None
        Q = None
        meta = None

    # Broadcast metadata and quantization matrix
    meta = comm.bcast(meta, root=0)
    Q = comm.bcast(Q, root=0)
    H_orig, W_orig, n_bh, n_bw, n_blocks, Kh, Kw = meta

    # Build DCT matrices on each rank (avoids transmitting large matrices)
    D_h = dct_matrix(Kh)
    D_w = dct_matrix(Kw)
    Di_h = idct_matrix(Kh)
    Di_w = idct_matrix(Kw)

    # --- Scatter blocks ---
    # Divide blocks as evenly as possible
    counts = [n_blocks // P + (1 if i < n_blocks % P else 0) for i in range(P)]
    local_n = counts[rank]
    local_blocks = np.zeros((local_n, Kh, Kw), dtype=np.float64)

    # Use Scatterv for uneven distribution
    if rank == 0:
        flat_send = blocks_flat.reshape(-1)
        block_size = Kh * Kw
        sendcounts = [c * block_size for c in counts]
        displacements = [sum(sendcounts[:i]) for i in range(P)]
    else:
        flat_send = None
        sendcounts = None
        displacements = None

    sendcounts_bcast = comm.bcast(sendcounts, root=0)
    displacements_bcast = comm.bcast(displacements, root=0)

    flat_recv = np.zeros(local_n * Kh * Kw, dtype=np.float64)
    comm.Scatterv([flat_send, sendcounts_bcast, displacements_bcast, MPI.DOUBLE],
                  [flat_recv, MPI.DOUBLE], root=0)
    local_blocks = flat_recv.reshape(local_n, Kh, Kw)

    t_after_scatter = MPI.Wtime()

    # --- Each rank processes its blocks ---
    local_rec = np.zeros_like(local_blocks)
    for i in range(local_n):
        block = local_blocks[i]
        Y = D_h @ block @ D_w.T
        Yq = np.round(Y / Q)
        Y_rec = Yq * Q
        local_rec[i] = Di_h @ Y_rec @ Di_w.T

    t_after_compute = MPI.Wtime()

    # --- Gather reconstructed blocks ---
    flat_local_rec = local_rec.reshape(-1)
    if rank == 0:
        flat_gathered = np.zeros(n_blocks * Kh * Kw, dtype=np.float64)
    else:
        flat_gathered = None

    comm.Gatherv([flat_local_rec, MPI.DOUBLE],
                 [flat_gathered, sendcounts_bcast, displacements_bcast, MPI.DOUBLE],
                 root=0)

    t_end = MPI.Wtime()

    # --- Rank 0: reconstruct image ---
    if rank == 0:
        rec_blocks = flat_gathered.reshape(n_bh, n_bw, Kh, Kw)
        rec_img = reconstruct_from_blocks(rec_blocks, H_orig, W_orig)
        rec_img = np.clip(rec_img + 128.0, 0, 255).astype(np.uint8)

        from PIL import Image
        Image.fromarray(rec_img).save("mpi_reconstructed.png")

        T_total = t_end - t_start
        T_scatter = t_after_scatter - t_start
        T_compute = t_after_compute - t_after_scatter
        T_gather = t_end - t_after_compute

        print(f"\nMPI Run: P={P}, block size={Kh}x{Kw}")
        print(f"  Scatter time:  {T_scatter:.4f} s")
        print(f"  Compute time:  {T_compute:.4f} s")
        print(f"  Gather time:   {T_gather:.4f} s")
        print(f"  Total time:    {T_total:.4f} s")
        print(f"  Saved: mpi_reconstructed.png")
        return rec_img, T_total


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_image.png"
    Kh = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    Kw = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    run_mpi_jpeg(image_path, Kh, Kw)
