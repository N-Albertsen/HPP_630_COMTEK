"""
JPEG DCT Compression - COMTEK 6 / ESD 6
Workshop Assignment - May 2026

Full implementation of JPEG lossy compression pipeline:
  Step 1: Conversion to grayscale / center pixel values
  Step 2: Block partitioning
  Step 3: 2D DCT (type-II)
  Step 4: Quantization
  Step 5: Reconstruction via inverse DCT (type-III)

Sequential methods:  NumPy vectorization, Numba JIT
Parallel methods:    Numba JIT (parallel=True), Multiprocessing

NOTE: scipy.fftpack is NOT used as per assignment requirements.
"""

import numpy as np
import math
import time
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import numba
from numba import njit, prange
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# JPEG Standard Quantization Matrices (8x8)
# Source: JPEG standard (ITU-T T.81)
# ─────────────────────────────────────────────────────────────────────────────

QUANT_MATRIX_8 = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float64)


def make_quantization_matrix(Kh: int, Kw: int) -> np.ndarray:
    """
    Generate a quantization matrix of arbitrary size Kh x Kw.
    For 8x8 the standard JPEG matrix is returned.
    For larger sizes, the 8x8 matrix is tiled and scaled so that
    higher-frequency positions get larger quantization steps
    (increasing compression at high frequencies), matching the
    philosophy of the standard JPEG matrix.
    """
    if Kh == 8 and Kw == 8:
        return QUANT_MATRIX_8.copy()

    # Build base frequency-scaled matrix using the same formula structure
    # as the standard: values increase toward the bottom-right (high freq)
    Q = np.zeros((Kh, Kw), dtype=np.float64)
    for i in range(Kh):
        for j in range(Kw):
            # Linearly interpolate quantization values based on frequency position
            # Lowest freq (0,0) -> ~16, highest freq -> ~100
            Q[i, j] = 1 + (i + j + 1) * (99 / (Kh + Kw - 2))
    # Ensure minimum value of 1
    Q = np.clip(Q, 1, None)
    return Q


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Image Conversion
# Complexity: O(N*M)  Work: O(N*M)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_convert(image_path: str) -> np.ndarray:
    """
    Load image, convert to grayscale (luma Y channel),
    and center pixel values around 0 by subtracting 128.
    Returns float64 array of shape (H, W).
    """
    img = Image.open(image_path).convert("L")   # grayscale
    arr = np.array(img, dtype=np.float64)
    arr -= 128.0   # center: range now [-128, 127]
    return arr


def load_and_convert_array(arr_rgb: np.ndarray) -> np.ndarray:
    """Same as above but accepts an RGB numpy array directly."""
    if arr_rgb.ndim == 3:
        # ITU-R BT.601 luma coefficients
        gray = (0.299 * arr_rgb[:, :, 0]
                + 0.587 * arr_rgb[:, :, 1]
                + 0.114 * arr_rgb[:, :, 2])
    else:
        gray = arr_rgb.astype(np.float64)
    return gray - 128.0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Block Partitioning
# Complexity: O(N*M / (Kh*Kw))  Work: O(N*M)
# ─────────────────────────────────────────────────────────────────────────────

def pad_image(img: np.ndarray, Kh: int, Kw: int) -> np.ndarray:
    """Pad image so its dimensions are multiples of Kh, Kw."""
    H, W = img.shape
    pad_h = (Kh - H % Kh) % Kh
    pad_w = (Kw - W % Kw) % Kw
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='edge')
    return img


def extract_blocks(img: np.ndarray, Kh: int, Kw: int):
    """
    Partition image into blocks of size Kh x Kw using stride tricks.
    Returns array of shape (n_blocks_h, n_blocks_w, Kh, Kw).
    """
    H, W = img.shape
    assert H % Kh == 0 and W % Kw == 0, "Image must be padded first"
    n_bh = H // Kh
    n_bw = W // Kw
    # Use reshape (O(1) view, no copy)
    blocks = img.reshape(n_bh, Kh, n_bw, Kw).swapaxes(1, 2)
    # shape: (n_bh, n_bw, Kh, Kw)
    return blocks


def reconstruct_from_blocks(blocks: np.ndarray, H_orig: int, W_orig: int) -> np.ndarray:
    """Reassemble blocks back to full image and crop to original size."""
    n_bh, n_bw, Kh, Kw = blocks.shape
    img = blocks.swapaxes(1, 2).reshape(n_bh * Kh, n_bw * Kw)
    return img[:H_orig, :W_orig]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 & 5: DCT and IDCT implementations
# Type-II DCT:  y_k = 2 * sum_{n=0}^{N-1} x_n * cos(pi*k*(2n+1)/(2N))
# Type-III IDCT: inverse of type-II
# ─────────────────────────────────────────────────────────────────────────────

# --- DCT basis matrix construction ---

def dct_matrix(N: int) -> np.ndarray:
    """
    Build the NxN DCT-II transformation matrix D such that Y = D @ x.
    D[k,n] = 2 * cos(pi*k*(2n+1)/(2N))
    Complexity of building: O(N^2)
    """
    k = np.arange(N).reshape(N, 1)
    n = np.arange(N).reshape(1, N)
    D = 2.0 * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    return D


def idct_matrix(N: int) -> np.ndarray:
    """
    Build the NxN DCT-III (inverse DCT) transformation matrix.
    Implements: x_n = (1/2N) * [ y_0 + 2*sum_{k=1}^{N-1} y_k*cos(pi*k*(2n+1)/(2N)) ]

    In matrix form:  Di[n,k]:
      - k=0:   Di[n,0] = 1/(2N)          (coefficient 1 for y_0 term)
      - k>=1:  Di[n,k] = 2*cos(...)/(2N)  (coefficient 2 for higher terms)

    This is NOT simply D.T/(2N) because the k=0 row of D has no factor of 2
    distinction — we must set the k=0 column of Di to 1/(2N), not 2*.../(2N).
    """
    k = np.arange(N).reshape(1, N)
    n = np.arange(N).reshape(N, 1)
    Di = 2.0 * np.cos(np.pi * k * (2 * n + 1) / (2 * N))
    Di[:, 0] = 1.0      # k=0: coefficient is 1, not 2
    Di = Di / (2.0 * N)
    return Di


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL METHOD 1: NumPy vectorized DCT
# ─────────────────────────────────────────────────────────────────────────────

class JPEGNumpy:
    """
    Full JPEG pipeline using explicit NumPy vectorization.
    Pre-computes DCT matrices for given block size.
    2D DCT = D @ block @ D.T  (separable transform)
    Work for one block: O(Kh*Kw*(Kh+Kw))
    Total work: O(N*M*(Kh+Kw)/(Kh*Kw)) = O(N*M/Kh + N*M/Kw)
    """

    def __init__(self, Kh: int, Kw: int):
        self.Kh = Kh
        self.Kw = Kw
        self.D_h = dct_matrix(Kh)      # (Kh x Kh)
        self.D_w = dct_matrix(Kw)      # (Kw x Kw)
        self.Di_h = idct_matrix(Kh)    # (Kh x Kh)
        self.Di_w = idct_matrix(Kw)    # (Kw x Kw)
        self.Q = make_quantization_matrix(Kh, Kw)
        
    def dct2d(self, block: np.ndarray) -> np.ndarray:
        """2D DCT: Y = D_h @ block @ D_w.T"""
        return self.D_h @ block @ self.D_w.T

    def idct2d(self, Y: np.ndarray) -> np.ndarray:
        """2D IDCT: block = Di_h @ Y @ Di_w.T"""
        return self.Di_h @ Y @ self.Di_w.T

    def quantize(self, Y: np.ndarray) -> np.ndarray:
        """Element-wise division by Q, round to nearest integer."""
        return np.round(Y / self.Q)

    def dequantize(self, Yq: np.ndarray) -> np.ndarray:
        """Multiply back by Q to reconstruct (lossy)."""
        return Yq * self.Q

    def compress_blocks(self, blocks: np.ndarray) -> np.ndarray:
        """
        Apply DCT + quantization to all blocks at once using broadcasting.
        blocks shape: (n_bh, n_bw, Kh, Kw)
        Returns quantized blocks of same shape.
        """
        # Vectorised 2D DCT over all blocks:
        # Y = D_h @ blocks @ D_w.T
        # Use einsum or matmul broadcasting
        Y = np.tensordot(blocks, self.D_w.T, axes=([3], [0]))   # (...,Kh,Kw)
        Y = np.tensordot(self.D_h, Y, axes=([1], [2]))           # (Kh,...,Kw) -> transpose
        Y = np.moveaxis(Y, 0, 2)                                  # (n_bh,n_bw,Kh,Kw)
        Yq = np.round(Y / self.Q)
        return Yq

    def reconstruct_blocks(self, Yq: np.ndarray) -> np.ndarray:
        """
        Dequantize + inverse DCT on all blocks.
        """
        Y = Yq * self.Q
        X = np.tensordot(Y, self.Di_w.T, axes=([3], [0]))
        X = np.tensordot(self.Di_h, X, axes=([1], [2]))
        X = np.moveaxis(X, 0, 2)
        return X

    def run(self, img: np.ndarray):
        """Full pipeline: compress + reconstruct. Returns reconstructed image."""
        H_orig, W_orig = img.shape
        img_pad = pad_image(img, self.Kh, self.Kw)
        blocks = extract_blocks(img_pad, self.Kh, self.Kw)
        Yq = self.compress_blocks(blocks)
        rec_blocks = self.reconstruct_blocks(Yq)
        rec_img = reconstruct_from_blocks(rec_blocks, H_orig, W_orig)
        # Undo centering and clip
        rec_img = np.clip(rec_img + 128.0, 0, 255).astype(np.uint8)
        return rec_img


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL METHOD 2: Numba JIT compiled DCT
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _dct1d_numba(x):
    """
    1D type-II DCT using the direct definition.
    y_k = 2 * sum_{n=0}^{N-1} x_n * cos(pi*k*(2n+1)/(2N))
    """
    N = len(x)
    y = np.zeros(N)
    for k in range(N):
        s = 0.0
        for n in range(N):
            s += x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
        y[k] = 2.0 * s
    return y


@njit(cache=True)
def _idct1d_numba(y):
    """
    1D type-III DCT (inverse).
    x_n = (1/(2N)) * [ y_0 + 2*sum_{k=1}^{N-1} y_k*cos(pi*k*(2n+1)/(2N)) ]
    """
    N = len(y)
    x = np.zeros(N)
    for n in range(N):
        s = y[0]
        for k in range(1, N):
            s += 2.0 * y[k] * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
        x[n] = s / (2.0 * N)
    return x


@njit(cache=True)
def _dct2d_numba(block):
    """2D DCT: apply 1D DCT along rows then columns."""
    Kh, Kw = block.shape
    tmp = np.zeros((Kh, Kw))
    out = np.zeros((Kh, Kw))
    # DCT along rows
    for i in range(Kh):
        tmp[i, :] = _dct1d_numba(block[i, :])
    # DCT along columns
    for j in range(Kw):
        out[:, j] = _dct1d_numba(tmp[:, j])
    return out


@njit(cache=True)
def _idct2d_numba(Y):
    """2D IDCT: apply 1D IDCT along columns then rows."""
    Kh, Kw = Y.shape
    tmp = np.zeros((Kh, Kw))
    out = np.zeros((Kh, Kw))
    for j in range(Kw):
        tmp[:, j] = _idct1d_numba(Y[:, j])
    for i in range(Kh):
        out[i, :] = _idct1d_numba(tmp[i, :])
    return out


@njit(cache=True)
def _compress_numba_seq(blocks_flat, n_bh, n_bw, Kh, Kw, Q):
    """
    Sequential Numba: compress all blocks (DCT + quantize).
    blocks_flat shape: (n_bh*n_bw, Kh, Kw)
    """
    n_blocks = n_bh * n_bw
    Yq = np.zeros_like(blocks_flat)
    for idx in range(n_blocks):
        Y = _dct2d_numba(blocks_flat[idx])
        for i in range(Kh):
            for j in range(Kw):
                Yq[idx, i, j] = round(Y[i, j] / Q[i, j])
    return Yq


@njit(cache=True)
def _reconstruct_numba_seq(Yq_flat, n_bh, n_bw, Kh, Kw, Q):
    """Sequential Numba: dequantize + IDCT for all blocks."""
    n_blocks = n_bh * n_bw
    out = np.zeros_like(Yq_flat)
    for idx in range(n_blocks):
        Y = np.zeros((Kh, Kw))
        for i in range(Kh):
            for j in range(Kw):
                Y[i, j] = Yq_flat[idx, i, j] * Q[i, j]
        out[idx] = _idct2d_numba(Y)
    return out


class JPEGNumbaSeq:
    """JPEG pipeline using sequential Numba JIT."""

    def __init__(self, Kh: int, Kw: int):
        self.Kh = Kh
        self.Kw = Kw
        self.Q = make_quantization_matrix(Kh, Kw)

    def run(self, img: np.ndarray) -> np.ndarray:
        H_orig, W_orig = img.shape
        img_pad = pad_image(img, self.Kh, self.Kw)
        blocks = extract_blocks(img_pad, self.Kh, self.Kw)
        n_bh, n_bw, Kh, Kw = blocks.shape
        blocks_flat = blocks.reshape(n_bh * n_bw, Kh, Kw).copy()

        Yq_flat = _compress_numba_seq(blocks_flat, n_bh, n_bw, Kh, Kw, self.Q)
        rec_flat = _reconstruct_numba_seq(Yq_flat, n_bh, n_bw, Kh, Kw, self.Q)

        rec_blocks = rec_flat.reshape(n_bh, n_bw, Kh, Kw)
        rec_img = reconstruct_from_blocks(rec_blocks, H_orig, W_orig)
        rec_img = np.clip(rec_img + 128.0, 0, 255).astype(np.uint8)
        return rec_img


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL METHOD 1: Numba JIT with parallel=True
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def _compress_numba_parallel(blocks_flat, n_bh, n_bw, Kh, Kw, Q):
    """
    Parallel Numba: each block processed in parallel using prange.
    """
    n_blocks = n_bh * n_bw
    Yq = np.zeros_like(blocks_flat)
    for idx in prange(n_blocks):
        Y = _dct2d_numba(blocks_flat[idx])
        for i in range(Kh):
            for j in range(Kw):
                Yq[idx, i, j] = round(Y[i, j] / Q[i, j])
    return Yq


@njit(parallel=True, cache=True)
def _reconstruct_numba_parallel(Yq_flat, n_bh, n_bw, Kh, Kw, Q):
    """Parallel Numba: each block reconstructed in parallel."""
    n_blocks = n_bh * n_bw
    out = np.zeros_like(Yq_flat)
    for idx in prange(n_blocks):
        Y = np.zeros((Kh, Kw))
        for i in range(Kh):
            for j in range(Kw):
                Y[i, j] = Yq_flat[idx, i, j] * Q[i, j]
        out[idx] = _idct2d_numba(Y)
    return out


class JPEGNumbaParallel:
    """JPEG pipeline using Numba JIT with parallel=True (OpenMP-like parallelism)."""

    def __init__(self, Kh: int, Kw: int):
        self.Kh = Kh
        self.Kw = Kw
        self.Q = make_quantization_matrix(Kh, Kw)

    def run(self, img: np.ndarray) -> np.ndarray:
        H_orig, W_orig = img.shape
        img_pad = pad_image(img, self.Kh, self.Kw)
        blocks = extract_blocks(img_pad, self.Kh, self.Kw)
        n_bh, n_bw, Kh, Kw = blocks.shape
        blocks_flat = blocks.reshape(n_bh * n_bw, Kh, Kw).copy()

        Yq_flat = _compress_numba_parallel(blocks_flat, n_bh, n_bw, Kh, Kw, self.Q)
        rec_flat = _reconstruct_numba_parallel(Yq_flat, n_bh, n_bw, Kh, Kw, self.Q)

        rec_blocks = rec_flat.reshape(n_bh, n_bw, Kh, Kw)
        rec_img = reconstruct_from_blocks(rec_blocks, H_orig, W_orig)
        rec_img = np.clip(rec_img + 128.0, 0, 255).astype(np.uint8)
        return rec_img


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL METHOD 2: Multiprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _process_chunk(args):
    """
    Worker function for multiprocessing.
    Processes a chunk of blocks: DCT + quantize + dequantize + IDCT.
    """
    chunk, Kh, Kw, Q = args
    D_h = dct_matrix(Kh)
    D_w = dct_matrix(Kw)
    Di_h = idct_matrix(Kh)
    Di_w = idct_matrix(Kw)

    results = np.zeros_like(chunk)
    for i in range(len(chunk)):
        block = chunk[i]
        Y = D_h @ block @ D_w.T
        Yq = np.round(Y / Q)
        Y_rec = Yq * Q
        rec = Di_h @ Y_rec @ Di_w.T
        results[i] = rec
    return results


class JPEGMultiprocessing:
    """
    JPEG pipeline using Python multiprocessing.
    Splits blocks across multiple processes.
    """

    def __init__(self, Kh: int, Kw: int, n_workers: int = None):
        self.Kh = Kh
        self.Kw = Kw
        self.Q = make_quantization_matrix(Kh, Kw)
        self.n_workers = n_workers or mp.cpu_count()

    def run(self, img: np.ndarray) -> np.ndarray:
        H_orig, W_orig = img.shape
        img_pad = pad_image(img, self.Kh, self.Kw)
        blocks = extract_blocks(img_pad, self.Kh, self.Kw)
        n_bh, n_bw, Kh, Kw = blocks.shape
        blocks_flat = blocks.reshape(n_bh * n_bw, Kh, Kw).copy()

        # Split blocks into chunks for each worker
        n_workers = min(self.n_workers, len(blocks_flat))
        chunks = np.array_split(blocks_flat, n_workers)
        args = [(chunk.copy(), Kh, Kw, self.Q) for chunk in chunks]

        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_process_chunk, args)

        rec_flat = np.concatenate(results, axis=0)
        rec_blocks = rec_flat.reshape(n_bh, n_bw, Kh, Kw)
        rec_img = reconstruct_from_blocks(rec_blocks, H_orig, W_orig)
        rec_img = np.clip(rec_img + 128.0, 0, 255).astype(np.uint8)
        return rec_img


# ─────────────────────────────────────────────────────────────────────────────
# TIMING UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def time_method(method_fn, img: np.ndarray, n_runs: int = 5):
    """
    Run method_fn(img) n_runs times and return (mean_time, std_time, result).
    First run may include JIT compilation overhead for Numba - reported separately.
    """
    times = []
    result = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = method_fn(img)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times)
    return times.mean(), times.std(), result


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE QUALITY EVALUATION (Task VI)
# ─────────────────────────────────────────────────────────────────────────────

def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR) in dB.
    Higher = better quality. PSNR > 40 dB is excellent.
    """
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def ssim_simple(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Simplified Structural Similarity Index (SSIM).
    Returns value in [-1, 1], closer to 1 is better.
    """
    orig = original.astype(np.float64)
    rec = reconstructed.astype(np.float64)
    mu_o, mu_r = orig.mean(), rec.mean()
    var_o = orig.var()
    var_r = rec.var()
    cov = np.mean((orig - mu_o) * (rec - mu_r))
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = (2 * mu_o * mu_r + C1) * (2 * cov + C2) / \
           ((mu_o ** 2 + mu_r ** 2 + C1) * (var_o + var_r + C2))
    return ssim


def quality_score(psnr_val: float) -> int:
    """
    Convert PSNR to subjective quality score 0-5 as required by task VI.
    Score 5: perfect  (PSNR > 40 dB)
    Score 0: horrible (PSNR < 20 dB)
    """
    if psnr_val == float('inf'):
        return 5
    elif psnr_val > 40:
        return 5
    elif psnr_val > 35:
        return 4
    elif psnr_val > 30:
        return 3
    elif psnr_val > 25:
        return 2
    elif psnr_val > 20:
        return 1
    else:
        return 0
