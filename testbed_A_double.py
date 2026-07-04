"""
testbed_A_double.py

Testbed A'' : a GENUINE geodesic on a CURVED manifold, crossing a model kink.

Purpose: close the hole that Testbed A' (a detour inside a full-D square) leaves.
A' shows a support-valid detour raises exposure, but a reviewer can object that a
uniform square has no nontrivial manifold, so the "detour" isn't a geodesic of any
learned data manifold. Here the data manifold is a real curved 1-D manifold, the
circle S^1, and the path b -> x is the actual geodesic ON the circle (the shorter
arc). The model has a kink line that the arc crosses. We check that the
regime-mixing remainder R and exposure E survive on a true geodesic, i.e. that
"on-manifold + geodesic" still does not imply "query-local".

Manifold:   M = S^1 (unit circle), points x = (cos t, sin t).
Model:      f(x1,x2) = ReLU(x1 - a) - 0.5 * (angle-linear feature)
            -> a vertical kink at x1 = a that the arc crosses.
            We keep it PWL so cell structure is exact.

Two paths from b (angle tb) to x (angle tx), both ENDPOINTS ON THE CIRCLE:
    (1) chord  : straight line in R^2 between b and x  (leaves the manifold)
    (2) geodesic: the shorter arc of S^1 between them  (stays ON the manifold)

For each we compute the honest line integral A_i = ∫ ∂_i f dz_i and the
decomposition A = L + R with L_i = ∂_i f(x) (x_i - b_i), plus exposure E_i using
the TV-correct definition E_i = ∫ |∂_i f(gamma) - ∂_i f(x)| |dgamma_i|.

NOTE ON f AT THE QUERY: ∂_i f(x) is the AMBIENT gradient of the chosen PWL
extension. On a thin manifold only the tangent differential is identifiable
(Theorem 1); here we deliberately fix one smooth ambient extension so that L is
well-defined and the comparison chord-vs-geodesic is about PATH, not about
extension ambiguity. That is the honest scope: this testbed is about regime
mixing (second geometry), not identifiability (first geometry).

GPU torch, double precision. You run it. I run nothing.
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


# ----- model: PWL, vertical kink at x1 = a --------------------------------
A_KINK = 0.0  # kink line x1 = 0, so it splits the circle into two arcs cleanly

def f_model(X):
    # X: (B,2).  f = ReLU(x1 - a) - 0.5 * x2   (PWL, kink at x1 = a)
    return torch.relu(X[:, 0] - A_KINK) - 0.5 * X[:, 1]


def batched_grad(f, X):
    X = X.clone().detach().requires_grad_(True)
    y = f(X)
    g = torch.autograd.grad(y.sum(), X)[0]
    return g.detach()


# ----- endpoints on the circle --------------------------------------------
def circle_pt(t):
    return torch.stack([torch.cos(t), torch.sin(t)], dim=-1)


def chord_path(b, x, n):
    t = (torch.arange(n, device=DEVICE, dtype=DTYPE) + 0.5) / n
    return b[None, :] + t[:, None] * (x - b)[None, :]


def geodesic_path(tb, tx, n, arc="short"):
    # arc="short": minor arc (wrap to [-pi,pi]); "long": major arc (2pi - short)
    dt_short = (tx - tb + torch.pi) % (2 * torch.pi) - torch.pi  # in [-pi,pi]
    if arc == "short":
        dt = dt_short
    else:
        # major arc: go the other way round the circle
        dt = dt_short - torch.sign(dt_short) * (2 * torch.pi)
    s = (torch.arange(n, device=DEVICE, dtype=DTYPE) + 0.5) / n
    angles = tb + s * dt
    return circle_pt(angles)


# ----- honest line-integral decomposition (TV-correct exposure) -----------
def audit(f, b, x, P):
    b = b.to(DEVICE, DTYPE); x = x.to(DEVICE, DTYPE)
    P = P.to(DEVICE, DTYPE)
    mids = 0.5 * (P[1:] + P[:-1])
    dz = P[1:] - P[:-1]                      # (T-1,2) signed segment displacement
    gx = batched_grad(f, x[None, :])[0]      # ambient grad at query
    gM = batched_grad(f, mids)               # (T-1,2)

    A = (gM * dz).sum(0)                       # line integral
    span = (x - b)
    L = gx * span                             # query-local response (endpoint span)
    R = A - L
    # TV-correct exposure: |grad diff| * |dz_i|
    E = ((gM - gx[None, :]).abs() * dz.abs()).sum(0)
    # TV of each coordinate along the path
    TV = dz.abs().sum(0)
    return {"A": A.cpu(), "L": L.cpu(), "R": R.cpu(), "E": E.cpu(),
            "TV": TV.cpu(), "grad_x": gx.cpu(), "span": span.cpu()}


def count_kink_crossings(P):
    # sign of (x1 - a) flipping along path
    s = torch.sign(P[:, 0].to(DEVICE, DTYPE) - A_KINK)
    return int((s[1:] != s[:-1]).sum().item())


def run(n=200000):
    print(f"device={DEVICE} dtype={DTYPE}  kink at x1={A_KINK}\n")
    # Endpoints close together on the BOTTOM of the circle, opposite sides of
    # the kink x1=0:  b at -3pi/4 (x1<0), x at -pi/4 (x1>0).
    # - chord: short straight segment across the bottom, x1 monotone, TV_1 small.
    # - geodesic SHORT: the minor arc along the bottom, also x1 ~monotone.
    # - geodesic LONG: the MAJOR arc up and over the top -> x1 sweeps
    #   -0.707 -> -1 -> +1 -> +0.707, crossing x1=0 TWICE, so TV_1 is large.
    #   This is the case where an on-manifold geodesic carries MORE remainder.
    tb = torch.tensor(-3 * torch.pi / 4, device=DEVICE, dtype=DTYPE)
    tx = torch.tensor(-1 * torch.pi / 4, device=DEVICE, dtype=DTYPE)
    b = circle_pt(tb)
    x = circle_pt(tx)
    print(f"b = {b.tolist()}  (x1<0 side)")
    print(f"x = {x.tolist()}  (x1>0 side)")
    print(f"grad f(x) ambient = {batched_grad(f_model, x[None,:])[0].tolist()}")
    print(f"|span| in x1 = {abs((x-b)[0].item()):.4f}\n")

    P_chord = chord_path(b, x, n)
    P_geo_s = geodesic_path(tb, tx, n, arc="short")
    P_geo_l = geodesic_path(tb, tx, n, arc="long")

    rows = [
        ("CHORD (leaves manifold)", P_chord),
        ("GEODESIC SHORT arc (on-manifold)", P_geo_s),
        ("GEODESIC LONG arc  (on-manifold)", P_geo_l),
    ]
    for name, P in rows:
        o = audit(f_model, b, x, P)
        xk = count_kink_crossings(P)
        print(f"[{name}]  kink crossings = {xk}")
        print(f"    A (attrib)  = {o['A'].tolist()}")
        print(f"    L (local)   = {o['L'].tolist()}   <- identical across paths")
        print(f"    R (mixing)  = {o['R'].tolist()}")
        print(f"    E (exposure)= {o['E'].tolist()}")
        print(f"    TV(gamma_i) = {o['TV'].tolist()}")
        print()

    print("-" * 60)
    print("WHAT TO CHECK:")
    print("  * All three cross the kink; SHORT/LONG geodesics stay ON S^1.")
    print("  * L identical everywhere (depends only on grad f(x)).")
    print("  * LONG geodesic: TV(gamma_1) >> chord TV(gamma_1) (crosses x1=0 twice),")
    print("    so by the TV bound |R_1| <= TV(gamma_1) * |jump| is LARGER-allowed;")
    print("    if |R_1|_long > |R_1|_chord, the on-manifold geodesic is WORSE.")
    print("  * If |R_1| stays similar but E_1 grows, the extra travel cancelled:")
    print("    that is the exposure/cancellation distinction, still non-local.")


if __name__ == "__main__":
    run()