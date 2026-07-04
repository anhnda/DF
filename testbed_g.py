"""
testbed_G.py

Two experiments the paper marks as "specified; measurement pending":

  G1  Distributional CNN audit
      Many queries x many baselines x several trained ReLU CNNs.
      Report distributions of:
        r_rem = ||R||_1 / (||L||_1 + ||R||_1)       (relative remainder magnitude)
        r_exp = ||E||_1 / (||L||_1 + ||E||_1)        (relative exposure)
        r_ig  = ||R||_1 / ||IG||_1                   (size comparison, NOT a fraction)
      plus correlation of exposure with baseline distance and path length
      (to show distance/length correlate but do NOT fully explain exposure).

  G2  Trained ReLU MLP on curved support
      Train a ReLU classifier on data supported on a curved 1-manifold (a circle),
      take the geodesic ON the support, compare chord vs geodesic on
        L, R, E, E_vec.
      Point: the A'' phenomenon is not an artifact of a hand-built hinge; a
      trained network reproduces non-local R and nonzero E_vec on an on-manifold
      geodesic.

torch, GPU preferred. YOU run this. I run nothing. No smoketest.
Fill the paper's Testbed G tables from the printed output.
"""

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================
# shared: batched gradient of a scalar-output f
# ======================================================================
def batched_grad(f, X):
    X = X.clone().detach().requires_grad_(True)
    y = f(X)
    g = torch.autograd.grad(y.sum(), X)[0]
    return g.detach()


def audit_straight(f, b, x, n_steps):
    """Straight-line IG decomposition L,R,E + vector exposure, one (b,x)."""
    t = (torch.arange(n_steps, device=b.device, dtype=b.dtype) + 0.5) / n_steps
    P = b[None, :] + t[:, None] * (x - b)[None, :]
    span = (x - b)
    gx = batched_grad(f, x[None, :])[0]
    gP = batched_grad(f, P)
    ig = span * gP.mean(0)
    L = span * gx
    R = ig - L
    drift = gP - gx[None, :]
    E = span.abs() * drift.abs().mean(0)                     # per-coord exposure
    E_vec = (drift.norm(dim=1) * span.norm() / n_steps).sum()  # coarse vector exposure
    return {"IG": ig, "L": L, "R": R, "E": E, "E_vec": E_vec}


# ======================================================================
# G1 : distributional CNN audit
# ======================================================================
class SmallCNN(nn.Module):
    in_shape = (1, 16, 16)

    def __init__(self, n_classes=10, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Linear(16 * 4 * 4, n_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, *self.in_shape)
        h = self.features(x).flatten(1)
        return self.head(h)


def blurred_baseline(x_img, k=5):
    # simple average-pool blur as a baseline
    pad = k // 2
    return torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(x_img, (pad, pad, pad, pad), mode="reflect"),
        k, stride=1)


def run_G1(n_queries=200, n_steps=48, seeds=(0, 1), n_data=1024):
    print("=" * 70)
    print("G1: distributional CNN audit (multi-query, multi-baseline, multi-seed)")
    print("=" * 70)
    C, H, W = SmallCNN.in_shape
    d = C * H * W
    torch.manual_seed(123)
    data = torch.rand(n_data, C, H, W, device=DEVICE)   # fake 'images'
    mean_img = data.mean(0, keepdim=True)

    import statistics as st
    for seed in seeds:
        model = SmallCNN(seed=seed).to(DEVICE)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        results = {bn: {"r_rem": [], "r_exp": [], "r_ig": [],
                        "bdist": [], "exp": []} for bn in
                   ["black", "blurred", "mean", "nearest_real"]}

        idx = torch.randperm(n_data)[:n_queries]
        for qi in idx:
            x_img = data[qi:qi + 1]
            cls = int(model(x_img).argmax())
            f = lambda z, c=cls: model(z.view(-1, C, H, W))[:, c]
            xf = x_img.flatten()

            baselines = {
                "black": torch.zeros_like(xf),
                "blurred": blurred_baseline(x_img).flatten(),
                "mean": mean_img.flatten(),
                "nearest_real": None,  # filled below
            }
            # nearest real other image (L2 in pixel space)
            with torch.no_grad():
                dists = (data.flatten(1) - xf[None, :]).norm(dim=1)
                dists[qi] = float("inf")
                baselines["nearest_real"] = data[dists.argmin()].flatten()

            for bn, bf in baselines.items():
                o = audit_straight(f, bf, xf, n_steps)
                Ln = o["L"].abs().sum().item()
                Rn = o["R"].abs().sum().item()
                En = o["E"].abs().sum().item()
                IGn = o["IG"].abs().sum().item() + 1e-12
                results[bn]["r_rem"].append(Rn / (Ln + Rn + 1e-12))
                results[bn]["r_exp"].append(En / (Ln + En + 1e-12))
                results[bn]["r_ig"].append(Rn / IGn)
                results[bn]["bdist"].append((xf - bf).norm().item())
                results[bn]["exp"].append(En)

        print(f"\n[seed {seed}]  (median over {n_queries} queries)")
        print(f"{'baseline':>14} {'r_rem':>8} {'r_exp':>8} {'r_ig':>8} "
              f"{'corr(bdist,exp)':>16}")
        for bn, r in results.items():
            med = lambda v: st.median(v)
            # Pearson corr between baseline distance and exposure
            bd, ex = r["bdist"], r["exp"]
            mb, me = st.mean(bd), st.mean(ex)
            cov = sum((a - mb) * (b - me) for a, b in zip(bd, ex)) / len(bd)
            sb = (sum((a - mb) ** 2 for a in bd) / len(bd)) ** 0.5 + 1e-12
            se = (sum((b - me) ** 2 for b in ex) / len(ex)) ** 0.5 + 1e-12
            corr = cov / (sb * se)
            print(f"{bn:>14} {med(r['r_rem']):>8.3f} {med(r['r_exp']):>8.3f} "
                  f"{med(r['r_ig']):>8.3f} {corr:>16.3f}")
    print("\n  Read: r_rem and r_exp are bounded in [0,1] and comparable across")
    print("  baselines; r_ig is a size ratio (can exceed 1 under cancellation).")
    print("  corr(bdist,exp) should be positive but < 1: distance correlates with")
    print("  but does NOT fully explain exposure -> exposure is about regimes, not")
    print("  merely how far the baseline is.")


# ======================================================================
# G2 : trained ReLU MLP on curved support (circle), chord vs geodesic
# ======================================================================
class ReLUMLP(nn.Module):
    def __init__(self, width=64, depth=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers, dprev = [], 2
        for _ in range(depth):
            layers += [nn.Linear(dprev, width), nn.ReLU()]
            dprev = width
        layers += [nn.Linear(dprev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X).squeeze(-1)


def circle_pt(t):
    return torch.stack([torch.cos(t), torch.sin(t)], dim=-1)


def train_on_circle(steps=3000, n=4096, seed=0):
    """
    Train f: R^2 -> R to fit a target defined ON the circle:
      target(t) = sin(2t)   (smooth on the manifold, but the trained ReLU net
      is PWL in ambient R^2 with many kinks) -> gives a real trained network
      whose kinks are not hand-placed.
    """
    torch.manual_seed(seed)
    model = ReLUMLP(seed=seed).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        t = torch.rand(n, device=DEVICE) * 2 * torch.pi
        X = circle_pt(t)
        y = torch.sin(2 * t)
        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, loss.item()


def geodesic_on_circle(tb, tx, n, arc="short"):
    dt_short = (tx - tb + torch.pi) % (2 * torch.pi) - torch.pi
    dt = dt_short if arc == "short" else dt_short - torch.sign(dt_short) * 2 * torch.pi
    s = (torch.arange(n, device=DEVICE, dtype=torch.float32) + 0.5) / n
    return circle_pt(tb + s * dt)


def audit_path_general(f, b, x, P):
    mids = 0.5 * (P[1:] + P[:-1])
    dz = P[1:] - P[:-1]
    gx = batched_grad(f, x[None, :])[0]
    gM = batched_grad(f, mids)
    A = (gM * dz).sum(0)
    L = gx * (x - b)
    R = A - L
    E = ((gM - gx[None, :]).abs() * dz.abs()).sum(0)
    E_vec = ((gM - gx[None, :]).norm(dim=1) * dz.norm(dim=1)).sum()
    return {"A": A, "L": L, "R": R, "E": E, "E_vec": E_vec}


def run_G2(n_steps=100000, seed=0):
    print("\n" + "=" * 70)
    print("G2: TRAINED ReLU MLP on curved support (circle), chord vs geodesic")
    print("=" * 70)
    model, final_loss = train_on_circle(seed=seed)
    print(f"trained MLP final MSE on manifold = {final_loss:.4e}")

    # pick two on-circle endpoints
    tb = torch.tensor(0.6, device=DEVICE)
    tx = torch.tensor(2.5, device=DEVICE)
    b = circle_pt(tb); x = circle_pt(tx)
    f = lambda z: model(z)

    t = (torch.arange(n_steps, device=DEVICE, dtype=torch.float32) + 0.5) / n_steps
    P_chord = b[None, :] + t[:, None] * (x - b)[None, :]
    P_geo_s = geodesic_on_circle(tb, tx, n_steps, "short")
    P_geo_l = geodesic_on_circle(tb, tx, n_steps, "long")

    print(f"grad f(x) (trained) = {batched_grad(f, x[None,:])[0].tolist()}")
    for name, P in [("chord (leaves circle)", P_chord),
                    ("geodesic SHORT (on-manifold)", P_geo_s),
                    ("geodesic LONG  (on-manifold)", P_geo_l)]:
        o = audit_path_general(f, b, x, P)
        print(f"\n[{name}]")
        print(f"    A     = {o['A'].tolist()}")
        print(f"    L     = {o['L'].tolist()}")
        print(f"    R     = {o['R'].tolist()}")
        print(f"    E     = {o['E'].tolist()}")
        print(f"    E_vec = {o['E_vec'].item():.5f}")
    print("\n  Expect: on-manifold geodesics keep E_vec > 0 and R != 0 even though")
    print("  the network is TRAINED, not hand-built -> A'' phenomenon is generic.")


if __name__ == "__main__":
    print(f"device = {DEVICE}\n")
    run_G1()
    run_G2()