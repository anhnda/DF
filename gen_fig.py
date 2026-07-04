"""
Gen 2 hinh cho paper LVP. Chay tren torch GPU, ve bang matplotlib.
Xuat: fig_regions.png, fig_coeffield.png  (dpi 200, tight)
Chay: python gen_figs.py
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- test function ----------
def f(x):
    x1, x2 = x[..., 0], x[..., 1]
    return torch.relu(x1 - x2) * x2 + torch.relu(x1 + x2 - 1) - 0.5 * x1

# ---------- LVP: local value projection -> (w, residual, n_eff) ----------
def lvp(Y, X, x0, tau=0.15):                       # x0: (M,2)
    K = torch.exp(-((X[None] - x0[:, None])**2).sum(-1) / (2 * tau**2)).sqrt()
    W = []
    for m in range(len(x0)):
        A = torch.cat([torch.ones(len(X), 1, device=dev), X - x0[m]], 1) * K[m][:, None]
        beta = torch.linalg.lstsq(A, (Y * K[m])[:, None]).solution.squeeze()
        W.append(beta[1:])
    return torch.stack(W)

# ============================================================
# FIGURE 1: ban do 4 vung + kink set + cac diem case study
# ============================================================
def fig_regions():
    n = 600
    xs = torch.linspace(-0.5, 2.5, n, device=dev)
    gx, gy = torch.meshgrid(xs, xs, indexing='ij')
    grid = torch.stack([gx, gy], -1)
    x1, x2 = grid[..., 0], grid[..., 1]

    # region id: A=0, B=1, C=2, D=3
    g1 = (x1 > x2)
    g2 = (x1 + x2 > 1)
    region = torch.zeros_like(x1, dtype=torch.long)
    region[g1 & g2] = 0    # A
    region[g1 & ~g2] = 1   # B
    region[~g1 & g2] = 2   # C
    region[~g1 & ~g2] = 3  # D
    region = region.cpu().numpy().T   # transpose: imshow row=y

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    # mau nhat, phan biet 4 vung
    colors = ['#cfe8ff', '#ffe0cf', '#d6f0d0', '#eee0f0']   # A B C D
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    ax.imshow(region, origin='lower', extent=[-0.5, 2.5, -0.5, 2.5],
              cmap=cmap, aspect='equal', alpha=0.9, vmin=0, vmax=3)

    # kink lines
    t = np.linspace(-0.5, 2.5, 2)
    ax.plot(t, t, 'k-', lw=1.6, label=r'$x_1=x_2$')
    ax.plot(t, 1 - t, 'k--', lw=1.6, label=r'$x_1+x_2=1$')

    # region labels
    ax.text(1.9, 1.1, 'A', fontsize=15, weight='bold', ha='center')
    ax.text(1.7, 0.0, 'B', fontsize=15, weight='bold', ha='center')
    ax.text(0.0, 1.7, 'C', fontsize=15, weight='bold', ha='center')
    ax.text(-0.2, -0.2, 'D', fontsize=15, weight='bold', ha='center')

    # case-study points
    pts = {
        'C=(0.2,1.5)': (0.2, 1.5),
        'dead (-1.5,0.2)': (-1.5, 0.2),   # ngoai khung, chi minh hoa -> bo neu muon
        '(1.5,1.0)': (1.5, 1.0),
        '(0.5,0.5)': (0.5, 0.5),
    }
    for name, (px, py) in pts.items():
        if -0.5 <= px <= 2.5 and -0.5 <= py <= 2.5:
            ax.plot(px, py, 'o', ms=7, mfc='crimson', mec='k', mew=0.8, zorder=5)
            ax.annotate(name, (px, py), textcoords='offset points',
                        xytext=(6, 6), fontsize=8)
    # baseline (0,0)
    ax.plot(0, 0, 's', ms=7, mfc='white', mec='k', mew=1.2, zorder=5)
    ax.annotate('baseline (0,0)', (0, 0), textcoords='offset points',
                xytext=(6, -14), fontsize=8)
    # IG path baseline -> C (cat kink g2)
    ax.plot([0, 0.2], [0, 1.5], color='crimson', lw=1.3, ls=':', zorder=4)

    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
    ax.set_title('Four activation regions and the kink set $K$', fontsize=11)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig('fig_regions.png', dpi=200, bbox_inches='tight')
    print('saved fig_regions.png')

# ============================================================
# FIGURE 2: coefficient field d w1 / d x_{0,2}
#   plateau 1[x1>x2] + delta trough tren duong cheo
#   ve heatmap + telescoping (dien tich ranh bat bien theo tau)
# ============================================================
def fig_coeffield():
    # data that
    torch.manual_seed(0)
    X = 4 * torch.rand(400_000, 2, device=dev) - 1
    Y = f(X)

    # grid 31x31 tren [-0.5,2.5]^2 (khop paper)
    m = 31
    xs = torch.linspace(-0.5, 2.5, m, device=dev)
    gx, gy = torch.meshgrid(xs, xs, indexing='ij')
    g = torch.stack([gx, gy], -1).reshape(-1, 2)

    W = lvp(Y, X, g, tau=0.15).reshape(m, m, 2)
    w1 = W[..., 0]
    # sai phan theo x2 (truc thu 2 cua grid)
    dx2 = (xs[1] - xs[0]).item()
    dw1_dx2 = (w1[:, 1:] - w1[:, :-1]) / dx2         # (m, m-1)
    field = dw1_dx2.cpu().numpy()

    # telescoping: tich phan tung hang -> bac thang 1.5 -> 0.5
    telescope = (dx2 * dw1_dx2.sum(1)).cpu().numpy()   # (m,)
    x1_vals = xs.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.2),
                                   gridspec_kw={'width_ratios': [1.35, 1]})

    # --- heatmap coefficient field ---
    x2_mid = 0.5 * (xs[1:] + xs[:-1]).cpu().numpy()
    im = ax1.imshow(field.T, origin='lower', aspect='auto',
                    extent=[x1_vals[0], x1_vals[-1], x2_mid[0], x2_mid[-1]],
                    cmap='RdBu_r', vmin=-4, vmax=1.2)
    ax1.plot(x1_vals, x1_vals, 'k-', lw=1.2)          # x1=x2 (day ranh)
    tt = np.linspace(-0.5, 2.5, 2)
    ax1.plot(tt, 1 - tt, 'k--', lw=1.0)              # x1+x2=1
    ax1.set_xlim(-0.5, 2.5); ax1.set_ylim(-0.5, 2.5)
    ax1.set_xlabel(r'$x_1$'); ax1.set_ylabel(r'$x_2$')
    ax1.set_title(r'$\partial w_1/\partial x_{0,2}$  (LVP field)', fontsize=11)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label('plateau $\\approx1$ / trough $\\ll0$', fontsize=8)

    # --- telescoping curve ---
    ax2.plot(x1_vals, telescope, 'o-', ms=3.5, color='#1f4e8c', lw=1.3,
             label='LVP telescoped')
    ax2.axhline(1.5, color='gray', ls='--', lw=0.9)
    ax2.axhline(0.5, color='gray', ls='--', lw=0.9)
    ax2.axvline(1.5, color='crimson', ls=':', lw=1.0, label=r'step at $x_1=1.5$')
    ax2.text(2.5, 1.53, 'analytic 1.5', fontsize=8, ha='right', color='gray')
    ax2.text(2.5, 0.53, 'analytic 0.5', fontsize=8, ha='right', color='gray')
    ax2.set_xlabel(r'$x_1$')
    ax2.set_ylabel(r'$\int \partial w_1/\partial x_{0,2}\,dx_2$')
    ax2.set_title(r'$\tau$-invariant trough area', fontsize=11)
    ax2.set_ylim(0, 2.0)
    ax2.legend(loc='center right', fontsize=8)

    fig.tight_layout()
    fig.savefig('fig_coeffield.png', dpi=200, bbox_inches='tight')
    print('saved fig_coeffield.png')

if __name__ == '__main__':
    fig_regions()
    fig_coeffield()
    print('done. dat 2 file PNG canh lvp.tex roi compile.')