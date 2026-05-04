"""
COMTEK 6 / ESD 6 - JPEG DCT Workshop
Main benchmark script

Tasks covered:
  IV  - Implementation (4 methods: NumPy, Numba seq, Numba parallel, Multiprocessing)
  V   - Execution time plots for two image sizes + optimal process count
  VI  - Image quality evaluation for different compression factors
  VII - MPI speedup analysis (theoretical, since cluster not available locally)

Usage:
  python run_benchmark.py [image_path]

If no image path given, a synthetic test image is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import os
import sys
from PIL import Image

from jpeg_dct import (
    load_and_convert, load_and_convert_array, pad_image,
    psnr, ssim_simple, quality_score, time_method,
    JPEGNumpy, JPEGNumbaSeq, JPEGNumbaParallel, JPEGMultiprocessing,
    make_quantization_matrix, QUANT_MATRIX_8
)
from jpeg_mpi import mpi_speedup_analysis


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_test_image(H: int, W: int, path: str):
    """Generate a synthetic test image with gradients, shapes and textures."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Gradient background
    for i in range(H):
        img[i, :, 0] = int(255 * i / H)
    for j in range(W):
        img[:, j, 1] = int(255 * j / W)
    # Circles
    cy, cx = H // 2, W // 2
    for r in range(20, min(H, W) // 3, 30):
        for angle in np.linspace(0, 2*np.pi, 300):
            y = int(cy + r * np.sin(angle))
            x = int(cx + r * np.cos(angle))
            if 0 <= y < H and 0 <= x < W:
                img[y, x] = [255, 255, 0]
    # High frequency texture in bottom right
    img[H//2:, W//2:, 2] = np.tile(
        np.array([[0, 255], [255, 0]], dtype=np.uint8),
        (H // 4 + 1, W // 4 + 1)
    )[:H//2, :W//2]
    Image.fromarray(img).save(path)
    print(f"Generated test image: {path}  ({H}x{W})")
    return path


def load_image(path: str) -> np.ndarray:
    """Load image and return centered grayscale array."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float64) - 128.0
    return arr


def get_original_gray(path: str) -> np.ndarray:
    """Return original grayscale uint8 for quality comparison."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Warm up Numba (first call compiles; exclude from timing)
# ─────────────────────────────────────────────────────────────────────────────

def warmup_numba(Kh: int, Kw: int):
    print("  Warming up Numba JIT (compiling)...")
    tiny = np.zeros((Kh * 2, Kw * 2))
    JPEGNumbaSeq(Kh, Kw).run(tiny)
    JPEGNumbaParallel(Kh, Kw).run(tiny)
    print("  Numba warm-up done.")


# ─────────────────────────────────────────────────────────────────────────────
# TASK IV + V: Benchmark all methods
# ─────────────────────────────────────────────────────────────────────────────

def benchmark(img_path: str, Kh: int, Kw: int, n_runs: int = 5,
              n_workers_list: list = None, label: str = ""):
    """
    Run all 4 methods on the image and return timing results.
    Also sweeps number of multiprocessing workers to find optimum (Task V).
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {label}  |  Block size: {Kh}x{Kw}  |  Runs: {n_runs}")
    print(f"{'='*60}")

    img = load_image(img_path)
    H, W = img.shape
    print(f"Image size: {H}x{W}")

    warmup_numba(Kh, Kw)

    methods = {
        "NumPy (vectorized)":        JPEGNumpy(Kh, Kw),
        "Numba JIT (sequential)":    JPEGNumbaSeq(Kh, Kw),
        "Numba JIT (parallel)":      JPEGNumbaParallel(Kh, Kw),
        "Multiprocessing":           JPEGMultiprocessing(Kh, Kw),
    }

    results = {}
    for name, method in methods.items():
        t_mean, t_std, rec = time_method(method.run, img, n_runs)
        results[name] = {"mean": t_mean, "std": t_std, "result": rec}
        print(f"  {name:<30}  {t_mean:.4f} ± {t_std:.4f} s")

    # Find optimal number of processes for Multiprocessing (Task V)
    if n_workers_list is None:
        import multiprocessing
        max_cpu = multiprocessing.cpu_count()
        n_workers_list = [1, 2, 4, 8, max_cpu]
        n_workers_list = sorted(set([w for w in n_workers_list if w <= max_cpu]))

    print(f"\n  Optimal workers sweep for Multiprocessing:")
    worker_times = {}
    for nw in n_workers_list:
        method = JPEGMultiprocessing(Kh, Kw, n_workers=nw)
        t_mean, t_std, _ = time_method(method.run, img, n_runs=3)
        worker_times[nw] = t_mean
        print(f"    Workers={nw:<3}  {t_mean:.4f} s")

    results["worker_sweep"] = worker_times
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TASK VI: Quality evaluation for different compression factors
# ─────────────────────────────────────────────────────────────────────────────

def quality_evaluation(img_path: str, factors: list = None):
    """
    Evaluate reconstructed image quality for different compression factors.
    Factor multiplies the quantization matrix Q → higher factor = more compression.
    """
    if factors is None:
        factors = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

    original = get_original_gray(img_path)
    img = load_image(img_path)
    Kh, Kw = 8, 8

    print(f"\n{'='*60}")
    print("TASK VI: Quality Evaluation vs Compression Factor")
    print(f"{'='*60}")
    print(f"{'Factor':>8} | {'PSNR (dB)':>10} | {'SSIM':>8} | {'Score':>6}")
    print("-" * 40)

    quality_results = []
    reconstructed_imgs = {}

    for factor in factors:
        Q_scaled = QUANT_MATRIX_8 * factor
        Q_scaled = np.clip(Q_scaled, 1, None)

        class ScaledJPEG(JPEGNumpy):
            def __init__(self):
                super().__init__(8, 8)
                self.Q = Q_scaled

        method = ScaledJPEG()
        rec = method.run(img)
        p = psnr(original, rec)
        s = ssim_simple(original, rec)
        score = quality_score(p)

        quality_results.append({
            "factor": factor, "psnr": p, "ssim": s, "score": score
        })
        reconstructed_imgs[factor] = rec
        print(f"{factor:>8.2f} | {p:>10.2f} | {s:>8.4f} | {score:>6}/5")

    return quality_results, reconstructed_imgs, original


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_timing_comparison(results_small, results_large,
                            label_small, label_large,
                            Kh_small, Kh_large, save_path="timing_comparison.png"):
    """Plot timing for two image sizes and two block sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("JPEG DCT Execution Time Comparison", fontsize=14, fontweight='bold')

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336"]
    method_names = ["NumPy (vectorized)", "Numba JIT (sequential)",
                    "Numba JIT (parallel)", "Multiprocessing"]

    for ax, results, label, Kh in [
        (axes[0], results_small, label_small, Kh_small),
        (axes[1], results_large, label_large, Kh_large),
    ]:
        means = [results[m]["mean"] for m in method_names]
        stds  = [results[m]["std"]  for m in method_names]
        short_names = ["NumPy", "Numba Seq", "Numba ∥", "Multiproc"]
        bars = ax.bar(short_names, means, yerr=stds, capsize=5,
                      color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f"{label}\nBlock size: {Kh}×{Kh}", fontsize=11)
        ax.set_ylabel("Time (s)")
        ax.set_xlabel("Method")
        ax.grid(axis='y', alpha=0.4)

        # Annotate bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{mean:.3f}s", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_worker_sweep(worker_results_small, worker_results_large,
                       save_path="worker_sweep.png"):
    """Plot multiprocessing speedup vs number of workers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Multiprocessing: Optimal Number of Workers", fontsize=13)

    for ax, wr, title in [
        (axes[0], worker_results_small, "Small image (Kh=Kw=8)"),
        (axes[1], worker_results_large, "Large image (full-image block)")
    ]:
        workers = list(wr.keys())
        times = list(wr.values())
        speedups = [times[0] / t for t in times]
        ax2 = ax.twinx()
        ax.bar(range(len(workers)), times, color="#2196F3", alpha=0.7, label="Time (s)")
        ax2.plot(range(len(workers)), speedups, "ro-", label="Speedup", linewidth=2)
        ax.set_xticks(range(len(workers)))
        ax.set_xticklabels([str(w) for w in workers])
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Time (s)", color="#2196F3")
        ax2.set_ylabel("Speedup", color="red")
        ax.set_title(title)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_quality(quality_results, reconstructed_imgs, original,
                  save_path="quality_evaluation.png"):
    """Plot quality vs compression factor and show example images."""
    factors   = [r["factor"] for r in quality_results]
    psnrs     = [r["psnr"]   for r in quality_results]
    ssims     = [r["ssim"]   for r in quality_results]
    scores    = [r["score"]  for r in quality_results]
    show_factors = [0.25, 1.0, 4.0, 16.0]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig)
    fig.suptitle("JPEG Quality vs Compression Factor", fontsize=14, fontweight='bold')

    # PSNR plot
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.semilogx(factors, psnrs, "b.-", linewidth=2, markersize=8)
    ax1.axhline(40, color='g', linestyle='--', alpha=0.7, label='Score=5 (40dB)')
    ax1.axhline(30, color='y', linestyle='--', alpha=0.7, label='Score=3 (30dB)')
    ax1.axhline(20, color='r', linestyle='--', alpha=0.7, label='Score=1 (20dB)')
    ax1.set_xlabel("Compression Factor (log scale)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("PSNR vs Compression Factor")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.4)

    # Quality score
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.semilogx(factors, scores, "rs-", linewidth=2, markersize=8)
    ax2.set_xlabel("Compression Factor (log scale)")
    ax2.set_ylabel("Quality Score (0-5)")
    ax2.set_title("Subjective Quality Score vs Compression Factor")
    ax2.set_ylim(-0.5, 5.5)
    ax2.grid(True, alpha=0.4)
    for f, s, p in zip(factors, scores, psnrs):
        ax2.annotate(f"  {s}/5\n  ({p:.0f}dB)",
                     (f, s), fontsize=7, alpha=0.8)

    # Show images for selected factors
    for idx, factor in enumerate(show_factors):
        ax = fig.add_subplot(gs[1, idx])
        rec = reconstructed_imgs.get(factor, None)
        if rec is not None:
            ax.imshow(rec, cmap='gray', vmin=0, vmax=255)
            r = next((r for r in quality_results if r["factor"] == factor), None)
            if r:
                ax.set_title(f"Factor={factor}\nPSNR={r['psnr']:.1f}dB  Score={r['score']}/5",
                             fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_mpi_speedup(speedup_data, N, M, save_path="mpi_speedup.png"):
    """Plot theoretical MPI speedup curve."""
    if speedup_data is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Theoretical MPI Speedup Analysis (Task VII)\n"
                 f"Image={N}×{M}, R=200 Mbps", fontsize=12)

    workers = [d[0] for d in speedup_data]
    speedups = [d[2] for d in speedup_data]
    efficiency = [d[3] for d in speedup_data]

    ax = axes[0]
    ax.plot(workers, speedups, "b.-", linewidth=2, markersize=8, label="Achievable")
    ax.plot(workers, workers, "k--", alpha=0.5, label="Ideal (linear)")
    ax.set_xlabel("Number of Processes P")
    ax.set_ylabel("Speedup S(P)")
    ax.set_title("Speedup vs P")
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_xscale('log', base=2)

    ax = axes[1]
    ax.plot(workers, efficiency, "r.-", linewidth=2, markersize=8)
    ax.axhline(80, color='green', linestyle='--', alpha=0.7, label='80% efficiency')
    ax.set_xlabel("Number of Processes P")
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title("Efficiency vs P")
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # ---- Image setup ----
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}. Generating synthetic image.")
            img_path = None
    else:
        img_path = None

    # Two image sizes for Task V
    if img_path is None:
        small_path = "output/test_small.png"
        large_path = "output/test_large.png"
        make_test_image(256, 256, small_path)
        make_test_image(512, 512, large_path)
    else:
        small_path = img_path
        # Make a larger version by tiling
        large_path = "output/test_large.png"
        orig = np.array(Image.open(small_path).convert("RGB"))
        H, W = orig.shape[:2]
        if H < 400:
            large = np.tile(orig, (2, 2, 1))
        else:
            large = orig
        Image.fromarray(large).save(large_path)
        print(f"Large image: {large_path} ({large.shape[0]}x{large.shape[1]})")

    N_RUNS = 5

    print("\n" + "="*60)
    print("COMTEK 6 / ESD 6 — JPEG DCT BENCHMARK")
    print("="*60)

    # ─────────────────────────────────────────────────────────────
    # BLOCK SIZE 1: Standard JPEG (Kh=Kw=8)
    # ─────────────────────────────────────────────────────────────
    print("\n[1/4] Standard JPEG block size (8×8) on small image...")
    r_small_8 = benchmark(small_path, 8, 8, n_runs=N_RUNS,
                           label=f"Small ({os.path.basename(small_path)})")

    print("\n[2/4] Standard JPEG block size (8×8) on large image...")
    r_large_8 = benchmark(large_path, 8, 8, n_runs=N_RUNS,
                           label=f"Large ({os.path.basename(large_path)})")

    # ─────────────────────────────────────────────────────────────
    # BLOCK SIZE 2: Full-image block (Kh=N, Kw=M)
    # ─────────────────────────────────────────────────────────────
    small_img = load_image(small_path)
    N_s, M_s = small_img.shape
    large_img = load_image(large_path)
    N_l, M_l = large_img.shape

    print(f"\n[3/4] Full-image block ({N_s}×{M_s}) on small image...")
    r_small_full = benchmark(small_path, N_s, M_s, n_runs=2,
                              label=f"Small full-block ({N_s}×{M_s})")

    print(f"\n[4/4] Full-image block ({N_l}×{M_l}) on large image...")
    r_large_full = benchmark(large_path, N_l, M_l, n_runs=2,
                              label=f"Large full-block ({N_l}×{M_l})")

    # ─────────────────────────────────────────────────────────────
    # PLOTS (Task V)
    # ─────────────────────────────────────────────────────────────
    print("\n[Plotting] Timing comparison plots...")
    plot_timing_comparison(
        r_small_8, r_large_8,
        f"Small image ({N_s}×{M_s})", f"Large image ({N_l}×{M_l})",
        8, 8,
        save_path="output/timing_block8.png"
    )
    plot_timing_comparison(
        r_small_full, r_large_full,
        f"Small full-block ({N_s}×{M_s})", f"Large full-block ({N_l}×{M_l})",
        N_s, N_l,
        save_path="output/timing_full_block.png"
    )
    plot_worker_sweep(r_small_8["worker_sweep"], r_large_8["worker_sweep"],
                      save_path="output/worker_sweep.png")

    # ─────────────────────────────────────────────────────────────
    # TASK VI: Quality evaluation
    # ─────────────────────────────────────────────────────────────
    print("\n[Task VI] Quality evaluation...")
    factors = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    qresults, qimgs, orig_gray = quality_evaluation(small_path, factors)
    plot_quality(qresults, qimgs, orig_gray, save_path="output/quality_evaluation.png")

    # ─────────────────────────────────────────────────────────────
    # TASK VII: MPI speedup analysis
    # ─────────────────────────────────────────────────────────────
    print("\n[Task VII] MPI theoretical speedup analysis...")
    # T_seq = Numba JIT sequential time: this is the explicit-loop sequential
    # baseline that MPI would replace on a distributed cluster. NumPy is already
    # BLAS-vectorised internally and harder to distribute; Numba sequential
    # represents a clean single-threaded implementation.
    T_seq_baseline = r_small_8["Numba JIT (sequential)"]["mean"]
    speedup_data = mpi_speedup_analysis(N_s, M_s, 8, 8, T_seq_baseline, P_max=64)
    plot_mpi_speedup(speedup_data, N_s, M_s, save_path="output/mpi_speedup.png")

    print("\n" + "="*60)
    print("ALL TASKS COMPLETE. Output files in ./output/")
    print("="*60)
    print("  output/timing_block8.png        — Task V (8x8 blocks)")
    print("  output/timing_full_block.png     — Task V (full-image blocks)")
    print("  output/worker_sweep.png          — Task V (optimal workers)")
    print("  output/quality_evaluation.png    — Task VI")
    print("  output/mpi_speedup.png           — Task VII")
    print()
    print("For MPI execution on a cluster:")
    print("  mpiexec -n 4 python jpeg_mpi.py <image_path> 8 8")
