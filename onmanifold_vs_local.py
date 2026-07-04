"""
onmanifold_vs_local.py

Claim being demonstrated:
    "A valid (on-manifold) path is not necessarily a local path."

Manifold-IG / geodesic-IG fix DATA-path geometry (gamma subset M).
They do NOT guarantee gamma stays inside the query's activation cell C_x.
So even a perfectly on-manifold path carries a regime-mixing remainder R
whenever it crosses a model kink.

Setup:
    model  f = ReLU(x1 + x2 - 1) - 0.5 x1     (kink line x1+x2=1)
    data manifold M : an arc that lies ON the data, but whose two endpoints
                      (baseline b and query x) sit on OPPOSITE sides of the kink.

We compare three paths b -> x:
    (1) straight line  (may leave M, standard IG)
    (2) on-manifold arc (geodesic-style, gamma subset M) -- STILL crosses kink
    (3) the ideal 'stay-in-cell' path (only exists if b already in C_x; here it can't)

For (1) and (2) we report IG, L (local), R (mixing), E (exposure), gate flips.
The point: R and E are NON-ZERO for BOTH, and L is identical for both,
because L depends only on grad f(x). On-manifold-ness changed nothing about
locality.

GPU torch, double precision. You run it. I don't.
"""

import torch
from regime_mixing_audit import (
    DEVICE, DTYPE, f_hinge, audit_path, count_gate_flips, batched_grad,
)


# ----------------------------------------------------------------------
# A curved on-manifold path between b and x:
# parametrize an arc that bulges away from the straight line but has the
# same endpoints. This mimics a geodesic on a curved data manifold M.
# gamma(t) = (1-t) b + t x + bulge * sin(pi t) * n_hat
#   with n_hat a unit normal to (x-b). Every point is a legitimate on-support
#   point of the uniform-on-[-1,3]^2 data (M is full-D here so 'on-manifold'
#   just means 'inside support' -- but the arc still crosses the kink).
# ----------------------------------------------------------------------
def arc_path(b, x, n_steps, bulge=0.8):
    b = b.to(DEVICE, DTYPE); x = x.to(DEVICE, DTYPE)
    t = (torch.arange(n_steps, device=DEVICE, dtype=DTYPE) + 0.5) / n_steps
    line = b[None, :] + t[:, None] * (x - b)[None, :]
    d = (x - b)
    n_hat = torch.tensor([-d[1], d[0]], device=DEVICE, dtype=DTYPE)
    n_hat = n_hat / n_hat.norm()
    offset = (bulge * torch.sin(torch.pi * t))[:, None] * n_hat[None, :]
    return line + offset, t


def path_integral_weights_from_points(P):
    """
    For a non-straight path, IG's line-integral must weight each segment by the
    displacement dz_i, not by dt. audit_path uses (x_i-b_i)*mean_t g_i, which is
    ONLY correct for straight lines. For the arc we do the honest line integral:
        A_i = sum_seg  g_i(midpoint) * (z_i^{k+1} - z_i^k)
    Returns midpoints (T-1,d) and per-segment dz (T-1,d).
    """
    mids = 0.5 * (P[1:] + P[:-1])
    dz = (P[1:] - P[:-1])
    return mids, dz


def audit_arbitrary_path(f, b, x, P):
    """
    Honest line-integral attribution for an arbitrary ordered path P (T,d),
    plus the SAME L/R/E decomposition. L depends only on grad f(x): it is
    path-independent by construction -> that's the whole point.
    """
    b = b.to(DEVICE, DTYPE); x = x.to(DEVICE, DTYPE)
    mids, dz = path_integral_weights_from_points(P.to(DEVICE, DTYPE))
    gx = batched_grad(f, x[None, :])[0]          # (d,)
    gM = batched_grad(f, mids)                    # (T-1,d)

    # true line integral A_i = sum g_i * dz_i
    A = (gM * dz).sum(dim=0)                       # (d,)

    # local reference L_i = ∂_i f(x) * (x_i - b_i)   (endpoint span, path-free)
    span = (x - b)
    L = gx * span

    R = A - L
    # exposure along THIS path: |∂_i f(z) - ∂_i f(x)| * |dz_i|, summed
    E = ((gM - gx[None, :]).abs() * dz.abs()).sum(dim=0)
    return {"A": A.cpu(), "L": L.cpu(), "R": R.cpu(), "E": E.cpu(),
            "grad_x": gx.cpu()}


def run_comparison(n_steps=40000):
    print("=" * 64)
    print("ON-MANIFOLD IS NOT LOCAL — straight vs on-manifold arc")
    print("=" * 64)
    b = torch.tensor([0.0, 0.0], dtype=DTYPE)
    x = torch.tensor([0.2, 1.5], dtype=DTYPE)

    # (1) straight line
    from regime_mixing_audit import straight_line
    P_line, _ = straight_line(b.to(DEVICE), x.to(DEVICE), n_steps)
    out_line = audit_arbitrary_path(f_hinge, b, x, P_line)
    flips_line = count_gate_flips(f_hinge, P_line)

    # (2) on-manifold arc (bulged, still endpoints on opposite sides of kink)
    P_arc, _ = arc_path(b, x, n_steps, bulge=0.8)
    out_arc = audit_arbitrary_path(f_hinge, b, x, P_arc)
    flips_arc = count_gate_flips(f_hinge, P_arc)

    def show(name, o, flips):
        print(f"\n[{name}]")
        print(f"  A  (attribution) = {o['A'].tolist()}")
        print(f"  L  (local, path-free) = {o['L'].tolist()}")
        print(f"  R  (regime mixing)    = {o['R'].tolist()}")
        print(f"  E  (exposure)         = {o['E'].tolist()}")
        print(f"  gate flips total      = {flips['total_flips']}")

    print(f"\ngrad_x ∇f(x) = {out_line['grad_x'].tolist()}  (true [0.5,1.0], identifiable)")
    show("STRAIGHT LINE  (standard IG)", out_line, flips_line)
    show("ON-MANIFOLD ARC (geodesic-style)", out_arc, flips_arc)

    print("\n" + "-" * 64)
    print("TAKEAWAY:")
    print("  * L is IDENTICAL for both paths (it only sees grad f(x)).")
    print("  * Both paths cross the kink (gate flips > 0) => both carry R != 0.")
    print("  * Making the path on-manifold changed the DATA geometry, NOT the")
    print("    query-locality. Regime-mixing remainder survives. QED-by-witness.")
    print("-" * 64)
    return out_line, out_arc


if __name__ == "__main__":
    print(f"device = {DEVICE}, dtype = {DTYPE}")
    run_comparison()
