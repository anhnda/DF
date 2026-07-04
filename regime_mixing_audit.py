"""
regime_mixing_audit.py

Exact decomposition of path attribution into
    A_i = L_i (query-local response) + R_i (regime-mixing remainder)
and an exposure term E_i that catches cancellation.

    IG_i(x;b) = (x_i - b_i) * d_i f(x)                      <- L_i  (local)
              + (x_i - b_i) * ∫_0^1 [d_i f(gamma(t)) - d_i f(x)] dt   <- R_i (mixing)

    E_i = |x_i - b_i| * ∫_0^1 | d_i f(gamma(t)) - d_i f(x) | dt        <- exposure

E_i >> |R_i| means the path DID cross other regimes but the signed remainder
cancelled -> the attribution is small NOT because it's local, but because
regimes triangulated to zero (the Shapley-≈0 pathology in Testbed A).

Everything is autograd-based so it works for arbitrary f: R^d -> R.
GPU by default. You run it. I don't.
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # double precision, like the paper's testbeds


# ----------------------------------------------------------------------
# gradient helper: d f / d x  at a batch of points, for scalar-output f
# ----------------------------------------------------------------------
def batched_grad(f, X):
    """
    f: callable R^{B x d} -> R^{B}   (per-row scalar output)
    X: (B, d) tensor
    returns grad: (B, d) with grad[b] = ∇ f(X[b])
    """
    X = X.clone().detach().requires_grad_(True)
    y = f(X)                      # (B,)
    assert y.dim() == 1, "f must return one scalar per row"
    g = torch.autograd.grad(y.sum(), X, create_graph=False)[0]
    return g.detach()


# ----------------------------------------------------------------------
# straight-line path gamma(t) = b + t (x - b)
# ----------------------------------------------------------------------
def straight_line(b, x, n_steps):
    """b,x: (d,)  ->  path points (n_steps, d), midpoint rule."""
    t = (torch.arange(n_steps, device=b.device, dtype=b.dtype) + 0.5) / n_steps
    return b[None, :] + t[:, None] * (x - b)[None, :], t


# ----------------------------------------------------------------------
# generic path attribution + decomposition for ONE query
# ----------------------------------------------------------------------
def audit_path(f, b, x, path_points, tstep_weight=None):
    """
    f:            callable (B,d)->(B,)
    b, x:         (d,) baseline and query
    path_points:  (T, d) sampled along the path (already ordered b -> x)
    tstep_weight: (T,) integration weights along t in [0,1]; default uniform 1/T

    Returns dict with per-coordinate:
        IG   : path integral attribution        (d,)
        L    : query-local response             (d,)
        R    : signed regime-mixing remainder    (d,)   ( IG - L )
        E    : exposure (unsigned drift * span)  (d,)
        grad_x : ∇f(x)                           (d,)
    """
    b = b.to(DEVICE, DTYPE)
    x = x.to(DEVICE, DTYPE)
    P = path_points.to(DEVICE, DTYPE)
    T = P.shape[0]
    if tstep_weight is None:
        w = torch.full((T,), 1.0 / T, device=DEVICE, dtype=DTYPE)
    else:
        w = tstep_weight.to(DEVICE, DTYPE)

    span = (x - b)                      # (d,)
    gx = batched_grad(f, x[None, :])[0] # ∇f(x)  (d,)
    gP = batched_grad(f, P)             # (T, d) grads along path

    # IG_i = (x_i-b_i) * mean_t d_i f(gamma(t))
    ig = span * (w[:, None] * gP).sum(dim=0)

    # L_i = (x_i-b_i) * d_i f(x)
    L = span * gx

    # R_i = IG_i - L_i  (== span * mean_t [ d_i f(path) - d_i f(x) ])
    R = ig - L

    # E_i = |x_i-b_i| * mean_t | d_i f(path) - d_i f(x) |
    drift = gP - gx[None, :]                        # (T,d)
    E = span.abs() * (w[:, None] * drift.abs()).sum(dim=0)

    return {
        "IG": ig.cpu(), "L": L.cpu(), "R": R.cpu(),
        "E": E.cpu(), "grad_x": gx.cpu(),
    }


# ----------------------------------------------------------------------
# PWL diagnostics: count activation-boundary crossings along the path
# (works for any f via sign changes of second difference of gradient;
#  for exact ReLU nets you can hook the masks instead — see note)
# ----------------------------------------------------------------------
def count_gate_flips(f, path_points, tol=1e-9):
    """
    Approx number of gradient-jump events along path (per coordinate + total).
    For a PWL net, ∇f is piecewise-constant; jumps between consecutive path
    samples mark activation-boundary crossings. Denser path_points -> tighter.
    """
    g = batched_grad(f, path_points.to(DEVICE, DTYPE))   # (T,d)
    dg = (g[1:] - g[:-1]).abs()                          # (T-1,d)
    flips_per_coord = (dg > tol).sum(dim=0).cpu()
    total_flips = (dg.max(dim=1).values > tol).sum().item()
    return {"flips_per_coord": flips_per_coord, "total_flips": total_flips}


# ======================================================================
# TESTBED A : pure ReLU hinge  f = ReLU(x1+x2-1) - 0.5 x1
#   query C=(0.2,1.5): true local slope in x1 = +0.5, gradient identifiable
#   baseline (0,0): whole box ON-SUPPORT (data uniform on [-1,3]^2)
#   expectation: IG1 wrong SIGN, but L1 = +0.5*(x1-b1) recovers local response
# ======================================================================
def f_hinge(X):
    # X: (B,2)
    return torch.relu(X[:, 0] + X[:, 1] - 1.0) - 0.5 * X[:, 0]


def run_testbed_A(n_steps=20000):
    print("=" * 60)
    print("TESTBED A — pure ReLU hinge (on-support regime-mixing)")
    print("=" * 60)
    b = torch.tensor([0.0, 0.0], dtype=DTYPE)
    x = torch.tensor([0.2, 1.5], dtype=DTYPE)

    P, t = straight_line(b.to(DEVICE), x.to(DEVICE), n_steps)
    out = audit_path(f_hinge, b, x, P)
    flips = count_gate_flips(f_hinge, P)

    span = (x - b)
    print(f"span (x-b)        = {span.tolist()}")
    print(f"grad_x  ∇f(x)     = {out['grad_x'].tolist()}   (true = [0.5, 1.0])")
    print(f"IG                = {out['IG'].tolist()}")
    print(f"L  (local resp.)  = {out['L'].tolist()}   <- (x_i-b_i)*∂f(x)")
    print(f"R  (regime mix)   = {out['R'].tolist()}   <- IG - L")
    print(f"E  (exposure)     = {out['E'].tolist()}")
    print(f"gate flips total  = {flips['total_flips']}  (>=1 => path crossed a regime)")
    print()
    print("READ THIS: coord 0 (x1):")
    print(f"  local slope ∂1 f(x) = {out['grad_x'][0].item():+.4f}  (== +0.5, well-defined)")
    print(f"  IG_1               = {out['IG'][0].item():+.4f}  (sign may flip / be wrong)")
    print(f"  L_1                = {out['L'][0].item():+.4f}  (the honest local response)")
    print(f"  R_1                = {out['R'][0].item():+.4f}  (mixing pushed IG off L)")
    print(f"  |R_1| vs E_1       = {out['R'][0].abs().item():.4f} vs {out['E'][0].item():.4f}")
    print("  -> if E_1 >> |R_1| on some feature, small IG hides cancellation, not locality")
    return out


if __name__ == "__main__":
    print(f"device = {DEVICE}, dtype = {DTYPE}\n")
    run_testbed_A()
