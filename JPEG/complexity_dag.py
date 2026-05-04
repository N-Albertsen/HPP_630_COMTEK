"""
COMTEK 6 / ESD 6 — JPEG DCT
Task I:  Work and Complexity Analysis
Task II: DAG descriptions for parallel implementation

This file documents the theoretical analysis. The DAG diagrams
are described textually here and are also auto-generated as
matplotlib figures when you run this script.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


# ═══════════════════════════════════════════════════════════════════════════
# TASK I: Work and Complexity
# ═══════════════════════════════════════════════════════════════════════════
"""
Image size: N x M pixels
Block size: Kh x Kw
Number of blocks: B = (N/Kh) * (M/Kw)

────────────────────────────────────────────────────────────────────────────
STEP 1: Conversion to grayscale + centering
────────────────────────────────────────────────────────────────────────────
  Work:       W₁ = O(N·M)
    - Each of the N·M pixels requires 3 multiplications + 2 additions (RGB→Y)
    - Plus 1 subtraction per pixel for centering
    - Total arithmetic ops: ~6·N·M
  Depth (parallel): D₁ = O(1)  [fully parallelisable over pixels]
  Span (critical path): O(1) with N·M processors

────────────────────────────────────────────────────────────────────────────
STEP 2: Block partitioning
────────────────────────────────────────────────────────────────────────────
  Work:       W₂ = O(N·M)
    - View/reshape operation in NumPy is O(1) (no data copy, just metadata)
    - Padding if needed: O(N·M)  (edge replication)
  Depth:      D₂ = O(1)

────────────────────────────────────────────────────────────────────────────
STEP 3: 2D DCT on all blocks
────────────────────────────────────────────────────────────────────────────
  Per block:
    - Row DCT:    Kw DCT-1D operations each of O(Kh·log Kh) for FFT-based
                  or O(Kh²) for direct definition
    - Column DCT: Kh DCT-1D operations each of O(Kw·log Kw) or O(Kw²)
    - Total per block (matrix method): O(Kh·Kw·(Kh + Kw))
    - Total per block (direct):        O(Kh²·Kw + Kh·Kw²) = O(Kh·Kw·(Kh+Kw))

  Total work (direct method):
    W₃ = B · O(Kh·Kw·(Kh+Kw))
       = (N·M)/(Kh·Kw) · O(Kh·Kw·(Kh+Kw))
       = O(N·M·(Kh+Kw))

  Special cases:
    - Kh = Kw = 8:      W₃ = O(16·N·M)   = O(N·M)  [constant factor]
    - Kh = N, Kw = M:   W₃ = O(N·M·(N+M)) [single large block, expensive!]

  Depth (parallel, all blocks independent):
    D₃ = O(Kh·Kw·(Kh+Kw))  [depth of one block; all blocks run in parallel]

────────────────────────────────────────────────────────────────────────────
STEP 4: Quantization
────────────────────────────────────────────────────────────────────────────
  Per block: Kh·Kw divisions + roundings
  Total work: W₄ = O(N·M)   [one op per pixel]
  Depth:      D₄ = O(1)     [all pixels independent]

────────────────────────────────────────────────────────────────────────────
STEP 5: Inverse DCT (type-III) reconstruction
────────────────────────────────────────────────────────────────────────────
  Same complexity as Step 3 (IDCT is the transpose of DCT matrix divided
  by 2N — same number of multiplications and additions):
  W₅ = O(N·M·(Kh+Kw))
  D₅ = O(Kh·Kw·(Kh+Kw))

────────────────────────────────────────────────────────────────────────────
SUMMARY TABLE
────────────────────────────────────────────────────────────────────────────
  Step | Work                  | Depth         | Parallelisable?
  ─────┼───────────────────────┼───────────────┼──────────────────
   1   | O(N·M)                | O(1)          | Yes, per-pixel
   2   | O(N·M)                | O(1)          | Yes, or O(1) view
   3   | O(N·M·(Kh+Kw))       | O(Kh·Kw·(K)) | Yes, per-block
   4   | O(N·M)                | O(1)          | Yes, per-element
   5   | O(N·M·(Kh+Kw))       | O(Kh·Kw·(K)) | Yes, per-block
  ─────┴───────────────────────┴───────────────┴──────────────────
  Total work: dominated by steps 3 and 5: O(N·M·(Kh+Kw))
  With Kh=Kw=8: O(16·N·M) — linear in image size
  With Kh=N,Kw=M: O(N·M·(N+M)) — cubic for square images

Note: "Work" is the total number of operations if done sequentially.
      "Depth" is the span — the length of the critical path, determining
      the minimum time achievable with unlimited parallelism.
      Speedup ≤ Work / Depth (Brent's theorem).
"""


# ═══════════════════════════════════════════════════════════════════════════
# TASK II: DAG for parallel implementation
# ═══════════════════════════════════════════════════════════════════════════

def draw_dag(save_path="output/dag_parallel.png"):
    """
    Draw Directed Acyclic Graphs illustrating parallel JPEG pipeline.

    DAG structure:
    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 1: Pixel conversion  (N·M independent nodes in parallel)   │
    │   p(0,0) → p(0,1) → ... → p(N,M)   [all independent]          │
    │                   ↓                                             │
    │ STEP 2: Block partition   (view: O(1), shown as single node)    │
    │                   ↓                                             │
    │ STEP 3+4+5: For each block B_ij (independent subgraph):        │
    │   B_ij → DCT_rows → DCT_cols → Quantize → IDCT_cols → IDCT_rows│
    │   All blocks B_ij can run in parallel                           │
    │                   ↓ (after all blocks done: synchronisation)   │
    │ STEP 5 (post): Reassemble image (single sequential node)       │
    └─────────────────────────────────────────────────────────────────┘
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Task II: DAGs for Parallel JPEG DCT Implementation",
                 fontsize=13, fontweight='bold')

    # ─── LEFT: High-level pipeline DAG ───
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.set_title("High-Level Pipeline DAG", fontsize=11)
    ax.axis('off')

    def box(ax, x, y, w, h, label, color='lightblue', fontsize=9):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, wrap=True,
                multialignment='center')

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2 + 0.35), xytext=(x1, y1 - 0.35),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

    # Nodes
    nodes = [
        (5, 10,   "Step 1: Pixel Conversion\n(N·M parallel ops)", 'lightyellow'),
        (5, 8.5,  "Step 2: Block Partitioning\n(O(1) view operation)", 'lightyellow'),
        (2, 6.5,  "Block B₀₀\nDCT", 'lightgreen'),
        (4, 6.5,  "Block B₀₁\nDCT", 'lightgreen'),
        (6, 6.5,  "Block B₁₀\nDCT", 'lightgreen'),
        (8, 6.5,  "⋯ Block Bᵢⱼ\nDCT", 'lightgreen'),
        (2, 5.0,  "Block B₀₀\nQuantize", '#FFD580'),
        (4, 5.0,  "Block B₀₁\nQuantize", '#FFD580'),
        (6, 5.0,  "Block B₁₀\nQuantize", '#FFD580'),
        (8, 5.0,  "⋯\nQuantize", '#FFD580'),
        (2, 3.5,  "Block B₀₀\nIDCT", 'lightcoral'),
        (4, 3.5,  "Block B₀₁\nIDCT", 'lightcoral'),
        (6, 3.5,  "Block B₁₀\nIDCT", 'lightcoral'),
        (8, 3.5,  "⋯\nIDCT", 'lightcoral'),
        (5, 1.8,  "Reassemble Image\n(synchronisation barrier)", 'lightgray'),
        (5, 0.5,  "Output Image", 'white'),
    ]

    for x, y, label, color in nodes:
        box(ax, x, y, 2.8 if y > 8 else 1.5, 0.9, label, color)

    # Arrows from step 1->2
    arrow(ax, 5, 10, 5, 8.5)

    # Step 2 -> blocks
    for bx in [2, 4, 6, 8]:
        ax.annotate("", xy=(bx, 6.5 + 0.45), xytext=(5, 8.5 - 0.45),
                    arrowprops=dict(arrowstyle="->", color='gray',
                                   connectionstyle="arc3,rad=0", lw=1))

    # Block DCT -> Quantize -> IDCT
    for bx in [2, 4, 6, 8]:
        arrow(ax, bx, 6.5, bx, 5.0)
        arrow(ax, bx, 5.0, bx, 3.5)
        ax.annotate("", xy=(5, 1.8 + 0.45), xytext=(bx, 3.5 - 0.45),
                    arrowprops=dict(arrowstyle="->", color='gray',
                                   connectionstyle="arc3,rad=0", lw=1))

    arrow(ax, 5, 1.8, 5, 0.5)

    # Parallel annotation
    ax.annotate("← All blocks run in parallel →",
                xy=(5, 5.5), fontsize=9, ha='center', color='darkgreen',
                fontstyle='italic')
    ax.annotate("⊗ Synchronisation barrier",
                xy=(5, 2.4), fontsize=8, ha='center', color='gray')

    # ─── RIGHT: Detailed single-block DAG ───
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_title("Detailed DAG for One Block (Steps 3–5)", fontsize=11)
    ax2.axis('off')

    def box2(ax, x, y, w, h, label, color='lightblue', fontsize=8):
        rect = mpatches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='black', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
                multialignment='center')

    def arr2(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2 + 0.3), xytext=(x1, y1 - 0.3),
                    arrowprops=dict(arrowstyle="->", color='black', lw=1.2))

    # Nodes for a Kh=4, Kw=4 example block
    box2(ax2, 5, 11, 5, 0.7, "Input Block X  [Kh×Kw]", 'lightyellow')

    # Row DCTs (can be parallelised)
    for i, y_ in enumerate([9.5, 8.5, 7.5, 6.5]):
        box2(ax2, 5, y_, 3, 0.6, f"DCT row {i}  →  tmp[{i},:]", 'lightgreen', 8)
        arr2(ax2, 5, 11, 5, y_)

    ax2.text(8.5, 8.5, "← Kh rows\n   in parallel", fontsize=8,
             color='darkgreen', fontstyle='italic')

    # Column DCTs
    box2(ax2, 5, 5.2, 5, 0.7, "Synchronise (all rows done)", 'lightgray')
    for y_ in [9.5, 8.5, 7.5, 6.5]:
        arr2(ax2, 5, y_, 5, 5.2)

    for j, y_ in enumerate([4.0, 3.2, 2.4, 1.6]):
        box2(ax2, 5, y_, 3.2, 0.6, f"DCT col {j}  →  Y[:,{j}]", 'lightblue', 8)
        arr2(ax2, 5, 5.2, 5, y_)

    ax2.text(8.5, 2.8, "← Kw cols\n   in parallel", fontsize=8,
             color='darkblue', fontstyle='italic')

    box2(ax2, 5, 0.7, 5, 0.6, "Quantize Y / Q  →  Yq  (step 4)", '#FFD580')
    for y_ in [4.0, 3.2, 2.4, 1.6]:
        arr2(ax2, 5, y_, 5, 0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)
    draw_dag()
    print("\nComplexity summary printed in module docstring.")
    print("Run: python -c \"import complexity_dag; help(complexity_dag)\"")
    print("     to read the full Task I analysis.")
