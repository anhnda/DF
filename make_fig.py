#!/usr/bin/env python3
# =====================================================================
#  make_figs.py  --  sinh 2 hình cho main.tex
#    fig_twopanel.png  : Figure 1  (regime-mixing  |  non-identifiability)
#    fig_coeffield.png : Testbed B  (coefficient field + jump trough)
#
#  Compute bằng Torch, vẽ bằng matplotlib. CHẠY:  python make_figs.py
#  Chạy được cả trên CPU (dev='cpu') lẫn GPU (dev='cuda').
#  Không cần asset ngoài. Ghi ra 2 file .png cạnh script.
# =====================================================================

import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # không cần display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# ---------------------------------------------------------------------
#  Test functions (khớp main.tex)
# ---------------------------------------------------------------------
def f_hinge(x):                            # Testbed A: piecewise-affine
    x1, x2 = x[..., 0], x[..., 1]
    return torch.relu(x1 + x2 - 1) - 0.5 * x1

def f_gate(x):                             # Testbed B: gated-quadratic
    x1, x2 = x[..., 0], x[..., 1]
    return torch.relu(x1 - x2) * x2 + torch.relu(x1 + x2 - 1) - 0.5 * x1

# ---------------------------------------------------------------------
#  Certified Tangent-LVP  (bản đồng bộ với main.tex; ở đây chỉ cần
#  omega_T nên trả về đúng nó cho gọn khi vẽ field)
# ---------------------------------------------------------------------
def tangent_lvp_w(Y, X, x0, tau=0.15, q=None, gap=10.0):
    """Trả về chỉ omega_T (M,d) — đủ để vẽ coefficient field.
    q=None: tự chọn tangent dim từ eigen-gap; q=d cố định -> raw LVP."""
    M, d = x0.shape
    W = torch.empty(M, d, device=dev)
    # chunk theo query để không nổ RAM khi M lớn
    CH = 256
    for s in range(0, M, CH):
        xb = x0[s:s+CH]                                   # (b,d)
        K = torch.exp(-((X[None] - xb[:, None])**2).sum(-1) / (2 * tau**2))  # (b,N)
        for j in range(xb.shape[0]):
            w = K[j]; wsum = w.sum()
            dX = X - xb[j]
            mu = (w[:, None] * dX).sum(0) / wsum
            Xc = dX - mu
            Sig = (w[:, None, None] * (Xc[:, :, None] * Xc[:, None, :])).sum(0) / wsum
            evals, evecs = torch.linalg.eigh(Sig)
            evals = evals.flip(0); evecs = evecs.flip(1)
            if q is None:
                ratios = evals[:-1] / evals[1:].clamp_min(1e-30)
                qm = int(torch.argmax(ratios).item()) + 1 if d > 1 else 1
                qm = qm if ratios.max() > gap else d       # fallback: full-dim
            else:
                qm = q
            Uh = evecs[:, :qm]
            Z = dX @ Uh
            A = torch.cat([torch.ones(len(X), 1, device=dev), Z], 1) * w[:, None].sqrt()
            b = (Y * w.sqrt())[:, None]
            beta = torch.linalg.lstsq(A, b).solution.squeeze()
            g = beta[1:]
            W[s+j] = Uh @ g
    return W

# =====================================================================
#  FIGURE 1  --  two panels
# =====================================================================
def make_twopanel(path="fig_twopanel.png"):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    # ----- LEFT: ReLU hinge, regime-mixing -----
    # background: value of f_hinge on a grid (chỉ để thấy kink), Torch tính
    gx = torch.linspace(-0.6, 2.6, 400, device=dev)
    gy = torch.linspace(-0.6, 2.6, 400, device=dev)
    GX, GY = torch.meshgrid(gx, gy, indexing='xy')
    G = torch.stack([GX, GY], -1)
    Z = f_hinge(G).cpu().numpy()
    ext = [-0.6, 2.6, -0.6, 2.6]
    axL.imshow(Z, origin='lower', extent=ext, aspect='equal',
               cmap='RdBu_r', alpha=0.65)
    # kink line x1+x2=1
    xs = np.linspace(-0.6, 2.6, 2)
    axL.plot(xs, 1 - xs, 'k-', lw=1.6, label=r'kink $x_1+x_2=1$')
    # region labels
    axL.text(0.05, 0.05, r'$x_1{+}x_2<1$' + '\nslope in $x_1$: $-0.5$',
             fontsize=9, va='bottom')
    axL.text(1.55, 1.75, r'$x_1{+}x_2>1$' + '\nslope in $x_1$: $+0.5$',
             fontsize=9, va='top', ha='center')
    # IG path baseline (0,0) -> C
    C = (0.2, 1.5)
    axL.plot([0, C[0]], [0, C[1]], color='crimson', ls=':', lw=2.2)
    axL.scatter([0], [0], c='k', s=28, zorder=5)
    axL.scatter([C[0]], [C[1]], c='crimson', s=45, zorder=5)
    axL.annotate('baseline $(0,0)$', (0, 0), (0.15, -0.35), fontsize=9,
                 arrowprops=dict(arrowstyle='-', lw=0.6))
    axL.annotate(r'$C=(0.2,1.5)$', C, (0.35, 1.95), fontsize=9,
                 arrowprops=dict(arrowstyle='-', lw=0.6))
    axL.set_xlim(-0.6, 2.6); axL.set_ylim(-0.6, 2.6)
    axL.set_xlabel('$x_1$'); axL.set_ylabel('$x_2$')
    axL.set_title('Full-dimensional support: regime-mixing', fontsize=11)
    axL.legend(loc='lower right', fontsize=8, framealpha=0.9)

    # ----- RIGHT: circle, non-identifiability -----
    th = np.linspace(0, 2*np.pi, 400)
    axR.plot(np.cos(th), np.sin(th), 'k-', lw=1.6, label=r'data support $S^1$')
    x0 = np.array([1.0, 0.0])
    axR.scatter([x0[0]], [x0[1]], c='crimson', s=45, zorder=5)
    # tangent (vertical) and normal (horizontal) at (1,0)
    axR.annotate('', (1.0, 0.6), (1.0, -0.6),
                 arrowprops=dict(arrowstyle='<->', color='seagreen', lw=1.8))
    axR.text(1.06, 0.5, 'tangent\n(identified)', color='seagreen', fontsize=9)
    # two candidate normal slopes = two models f_c agreeing on the circle
    for c, col, dx in [(0.0, 'gray', 1.0), (0.6, 'darkorange', 1.55)]:
        # arrow along +x (normal dir) with length coding the ambiguous slope
        axR.annotate('', (dx, 0.0), (1.0, 0.0),
                     arrowprops=dict(arrowstyle='->', color=col, lw=1.8))
    axR.text(1.6, 0.12, 'normal\n(unidentified: any $c$)',
             color='darkorange', fontsize=9)
    axR.set_xlim(-1.4, 2.1); axR.set_ylim(-1.4, 1.4)
    axR.set_aspect('equal')
    axR.set_xlabel('$x_1$'); axR.set_ylabel('$x_2$')
    axR.set_title('Thin support: non-identifiability', fontsize=11)
    axR.legend(loc='lower left', fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] wrote {path}")

# =====================================================================
#  FIGURE 2  --  coefficient field + jump trough (Testbed B)
# =====================================================================
def make_coeffield(path="fig_coeffield.png", N=200_000, tau=0.15):
    # real data trên [-1,3]^2
    X = 4 * torch.rand(N, 2, device=dev) - 1
    Y = f_gate(X)                                        # gated-quadratic values

    # grid 31x31 trên [-0.5,2.5]^2 (khớp main.tex)
    n = 31
    ax = torch.linspace(-0.5, 2.5, n, device=dev)
    GX, GY = torch.meshgrid(ax, ax, indexing='ij')
    grid = torch.stack([GX, GY], -1).reshape(-1, 2)

    # coefficient field w(x0); dùng full-dim (q=2) vì support là 2D
    W = tangent_lvp_w(Y, X, grid, tau=tau, q=2).reshape(n, n, 2)
    w1 = W[..., 0]                                        # ∂f/∂x1 field

    # ∂w1/∂x2  (finite difference dọc trục x2 = chiều thứ 2 của grid)
    dw1_dx2 = (w1[:, 1:] - w1[:, :-1]) / (ax[1] - ax[0])
    dw1_dx2 = dw1_dx2.cpu().numpy()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # panel 1: w1 field
    im1 = a1.imshow(w1.cpu().numpy().T, origin='lower',
                    extent=[-0.5, 2.5, -0.5, 2.5], aspect='equal', cmap='viridis')
    xs = np.linspace(-0.5, 2.5, 2)
    a1.plot(xs, xs, 'w--', lw=1.2)                        # kink x1=x2
    a1.set_title(r'coefficient field $w_1(x_0)=\partial f/\partial x_1$', fontsize=11)
    a1.set_xlabel('$x_1$'); a1.set_ylabel('$x_2$')
    fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.04)

    # panel 2: ∂w1/∂x2 -- plateau ~1 một bên, trough dọc x1=x2
    im2 = a2.imshow(dw1_dx2.T, origin='lower',
                    extent=[-0.5, 2.5, -0.5, 2.5], aspect='equal', cmap='RdBu_r',
                    vmin=-4, vmax=2)
    a2.plot(xs, xs, 'k--', lw=1.2)
    a2.set_title(r'$\partial w_1/\partial x_2$: plateau $\mathbb{1}[x_1{>}x_2]$ + jump trough',
                 fontsize=10)
    a2.set_xlabel('$x_1$'); a2.set_ylabel('$x_2$')
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.04)

    # telescoping check: ∫ ∂w1/∂x2 dx2 mỗi hàng -> nên đi từ ~1.5 xuống ~0.5
    tele = (dw1_dx2.sum(axis=1) * (ax[1] - ax[0]).item())
    print("[coeffield] telescoping ∫ per row (kỳ vọng bước 1.5 -> 0.5):")
    print(np.array2string(tele, precision=3, max_line_width=120))

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] wrote {path}")

# =====================================================================
if __name__ == "__main__":
    print(f"device = {dev}")
    make_twopanel("fig_twopanel.png")
    make_coeffield("fig_coeffield.png")
    print("done. Đặt 2 file .png cạnh main.tex rồi compile.")