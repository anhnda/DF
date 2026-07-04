#!/usr/bin/env python3
# =====================================================================
#  run_p0.py  --  P0 controlled experiments (oracle dimension).
#  Validates the THEORY on closed-form testbeds. NOT full-method eval:
#  q is passed as an oracle; automatic geometry routing is P1.
#
#  Torch for all compute. RUN:  python run_p0.py
#  CPU or GPU (auto). Prints a pass/fail-style report; no files written.
#
#  Stop rule (from review): do not proceed to V-shape / noisy / real
#  data until A-D all pass here.
# =====================================================================
import math
import torch

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
DT = torch.float64                       # tight tolerances need double
torch.set_default_dtype(DT)

def _fmt(v):
    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()
    return v

# ---------------------------------------------------------------------
#  Core estimator (matches main.tex; returns everything we probe)
# ---------------------------------------------------------------------
def tangent_lvp(Y, X, x0, tau=0.15, q=None, gap=10.0, kernel='gauss', radius=None):
    """Certified Tangent-LVP, single or few queries.
    kernel: 'gauss' (infinite support) or 'compact' (hard cutoff at `radius`).
    Returns dict per query index.
    """
    M, d = x0.shape
    out = []
    for m in range(M):
        diff = X - x0[m]
        r2 = (diff**2).sum(-1)
        if kernel == 'gauss':
            w = torch.exp(-r2 / (2 * tau**2))
        elif kernel == 'compact':
            rad = radius if radius is not None else 3 * tau
            inside = r2 <= rad**2
            w = torch.exp(-r2 / (2 * tau**2)) * inside    # smooth * hard cutoff
        else:
            raise ValueError(kernel)
        wsum = w.sum()
        mu = (w[:, None] * diff).sum(0) / wsum
        Xc = diff - mu
        Sig = (w[:, None, None] * (Xc[:, :, None] * Xc[:, None, :])).sum(0) / wsum
        evals, evecs = torch.linalg.eigh(Sig)
        evals = evals.flip(0); evecs = evecs.flip(1)      # descending
        if q is None:
            ratios = evals[:-1] / evals[1:].clamp_min(1e-30)
            qm = int(torch.argmax(ratios).item()) + 1 if d > 1 else 1
            qm = qm if ratios.max() > gap else d          # isotropic -> full dim
        else:
            qm = q
        Uh = evecs[:, :qm]
        Z = diff @ Uh
        A = torch.cat([torch.ones(len(X), 1, device=dev), Z], 1) * w[:, None].sqrt()
        b = (Y * w.sqrt())[:, None]
        beta = torch.linalg.lstsq(A, b).solution.squeeze()
        g = beta[1:].reshape(-1)
        omega_T = Uh @ g
        resid = ((A @ beta.reshape(-1, 1) - b).pow(2).sum() / wsum).sqrt()
        out.append(dict(omega_T=omega_T, U=Uh, q=qm,
                        evals=evals, R=resid, Rn=resid / tau,
                        neff=(wsum**2 / w.pow(2).sum())))
    return out

# ---------------------------------------------------------------------
#  Reference counterfactual attributions (closed-form, for Testbed A)
# ---------------------------------------------------------------------
def integrated_gradients(f, x, b, steps=2000):
    """IG_i = (x_i - b_i) * ∫_0^1 ∂_i f(b + a (x-b)) da, trapezoid."""
    x = x.to(DT); b = b.to(DT)
    a = torch.linspace(0, 1, steps, device=dev).reshape(-1, 1)
    pts = b + a * (x - b)                                  # (steps, d)
    pts.requires_grad_(True)
    y = f(pts).sum()
    grad, = torch.autograd.grad(y, pts)
    avg = grad.mean(0)                                    # ∫ da
    return (x - b) * avg

def baseline_shapley(f, x, b):
    """Exact Shapley with single baseline b, interventional (2^d coalitions).
    d small (=2), enumerate. Value of coalition S: f(x_S, b_{~S})."""
    x = x.to(DT); b = b.to(DT); d = x.numel()
    from itertools import combinations, permutations
    idx = list(range(d))
    phi = torch.zeros(d, device=dev)
    perms = list(permutations(idx))
    for i in idx:
        contrib = 0.0
        for p in perms:
            before = p[:p.index(i)]
            def val(S):
                z = b.clone()
                for j in S: z[j] = x[j]
                return f(z.reshape(1, -1)).squeeze()
            contrib += (val(list(before) + [i]) - val(list(before)))
        phi[i] = contrib / len(perms)
    return phi

# =====================================================================
def testbed_A():
    print("\n=== Testbed A: pure ReLU hinge (regime-mixing, gradient IS identifiable) ===")
    def f(x):
        x1, x2 = x[..., 0], x[..., 1]
        return torch.relu(x1 + x2 - 1) - 0.5 * x1
    X = (4 * torch.rand(400_000, 2, device=dev) - 1).to(DT)
    C = torch.tensor([[0.2, 1.5]], device=dev, dtype=DT)
    b = torch.tensor([0.0, 0.0], device=dev, dtype=DT)

    # ground-truth gradient at C (x1+x2=1.7>1): (0.5, 1)
    gtrue = torch.tensor([0.5, 1.0], device=dev)
    print(f"  analytic grad f(C)          = {_fmt(gtrue)}   (identifiable, full-dim)")

    # Tangent-LVP, oracle q=2, compact kernel so residual is exactly 0
    # dist from C to hinge x1+x2=1: |0.2+1.5-1|/sqrt2 = 0.7/1.414 = 0.495
    d_hinge = abs(0.2 + 1.5 - 1) / math.sqrt(2)
    r = 0.6 * d_hinge                                     # radius below dist-to-kink
    res = tangent_lvp(f(X), X, C, tau=0.15, q=2, kernel='compact', radius=r)[0]
    print(f"  Tangent-LVP omega_T (q=2)   = {_fmt(res['omega_T'])}   (expect ~[0.5,1])")
    print(f"  residual (compact, r<dist)  = {float(res['R']):.2e}   (expect ~0)")

    # IG and Shapley (closed form)
    ig = integrated_gradients(f, C.reshape(-1), b)
    sh = baseline_shapley(f, C.reshape(-1), b)
    print(f"  IG(C; b=0)                  = {_fmt(ig)}   (feat 1 expect WRONG SIGN <0)")
    print(f"  BaselineShapley(C; b=0)     = {_fmt(sh)}   (feat 1 expect ~0)")
    print(f"  detector |IG-SHAP|          = {_fmt((ig-sh).abs())}   (>0: box crosses hinge)")

    # control baseline whose box stays on one side of the hinge (both coords >?)
    bc = torch.tensor([0.6, 1.4], device=dev, dtype=DT)   # box [0.2,0.6]x[1.4,1.5] all x1+x2>1
    igc = integrated_gradients(f, C.reshape(-1), bc)
    shc = baseline_shapley(f, C.reshape(-1), bc)
    print(f"  control |IG-SHAP| (one cell)= {_fmt((igc-shc).abs())}   (expect ~0)")

def testbed_B_line_metric():
    print("\n=== Testbed C: line support x1=x2 (non-identifiability + metric probe) ===")
    t = (4 * torch.rand(400_000, 1, device=dev) - 1).to(DT)
    Xline = torch.cat([t, t], 1)
    Y = t.squeeze()                                       # f_c|_M = t for any c
    x0 = torch.tensor([[0.0, 0.0]], device=dev, dtype=DT)

    res = tangent_lvp(Y, Xline, x0, tau=0.2, q=1)[0]
    print(f"  Tangent-LVP omega_T (q=1)   = {_fmt(res['omega_T'])}   (expect [0.5,0.5])")
    print(f"  tangent basis U             = {_fmt(res['U'].reshape(-1))}   (expect ~±[.707,.707])")
    print(f"  ambiguity set: (0.5,0.5) + span(1,-1)   [unbounded normal direction]")

    # ridge-metric probe: same on-line fit, different Q -> different representative
    # Solve min ||A w - y||^2 + lam w^T Q w  on the FULL ambient (q=d=2) design.
    tau = 0.2
    w0 = x0.reshape(-1)
    diff = Xline - w0
    wk = torch.exp(-(diff**2).sum(-1) / (2 * tau**2))
    A = diff * wk[:, None]                                # no intercept: slope only probe
    yv = (Y * wk)[:, None]
    lam = 1e-3
    for Q, tag in [(torch.diag(torch.tensor([1.0, 100.0], device=dev)), "diag(1,100)"),
                   (torch.diag(torch.tensor([100.0, 1.0], device=dev)), "diag(100,1)")]:
        wsol = torch.linalg.solve(A.T @ A + lam * Q, A.T @ yv).reshape(-1)
        print(f"  ridge rep Q={tag:11s}   = {_fmt(wsol)}   (tilts; fit on line identical)")

def testbed_C_circle():
    print("\n=== Testbed D: circle S^1, f=1-x1 (curvature leakage vs tangent fix) ===")
    th = (2 * math.pi * torch.rand(400_000, device=dev)).to(DT)
    X = torch.stack([th.cos(), th.sin()], 1)
    Y = 1 - X[:, 0]
    x0 = torch.tensor([[1.0, 0.0]], device=dev, dtype=DT)

    raw = tangent_lvp(Y, X, x0, tau=0.15, q=2)[0]         # raw ambient (q=d)
    tan = tangent_lvp(Y, X, x0, tau=0.15, q=1)[0]         # tangent (q=1)
    print(f"  raw ambient LVP  (q=2)      = {_fmt(raw['omega_T'])}   (expect ~[-1,0]: LEAKAGE)")
    print(f"  Tangent-LVP      (q=1)      = {_fmt(tan['omega_T'])}   (expect ~[0,0]: true tangent deriv)")
    print(f"  tangent basis U             = {_fmt(tan['U'].reshape(-1))}   (expect ~±[0,1])")

    # spectrum log-log: lambda_T ~ tau^2, lambda_N ~ tau^4
    print("  spectrum eigenscaling (expect slope 2 tangent, 4 normal):")
    taus = [0.2, 0.1, 0.05, 0.025]
    lamT, lamN = [], []
    for tau in taus:
        r = tangent_lvp(Y, X, x0, tau=tau, q=2)[0]
        ev = r['evals']
        lamT.append(float(ev[0])); lamN.append(float(ev[1]))
    import math as _m
    def slope(taus, lams):
        lx = [_m.log(t) for t in taus]; ly = [_m.log(max(v, 1e-30)) for v in lams]
        n = len(lx); mx = sum(lx)/n; my = sum(ly)/n
        return sum((a-mx)*(b-my) for a, b in zip(lx, ly)) / sum((a-mx)**2 for a in lx)
    print(f"    lambda_T = {['%.2e'%v for v in lamT]}  slope={slope(taus,lamT):.2f}")
    print(f"    lambda_N = {['%.2e'%v for v in lamN]}  slope={slope(taus,lamN):.2f}")

def testbed_D_residual():
    print("\n=== Residual scaling: certificate for smooth vs kink vs affine ===")
    X = (2 * torch.rand(400_000, 2, device=dev) - 1).to(DT)   # [-1,1]^2
    x0 = torch.tensor([[0.0, 0.0]], device=dev, dtype=DT)
    fns = {
        'affine  x1':        lambda x: x[..., 0],
        'smooth  x1^2+x2^2': lambda x: x[..., 0]**2 + x[..., 1]**2,
        'kink    ReLU(x1)':  lambda x: torch.relu(x[..., 0]),
    }
    taus = [0.2, 0.1, 0.05, 0.025]
    for name, f in fns.items():
        Y = f(X)
        Rn = []
        for tau in taus:
            # affine tested with compact kernel to show exact-zero
            kw = dict(kernel='compact', radius=2.5*tau) if name.startswith('affine') else dict()
            r = tangent_lvp(Y, X, x0, tau=tau, q=2, **kw)[0]
            Rn.append(float(r['Rn']))
        # slope of R (not R/tau) in log-log to read the exponent
        import math as _m
        R = [rn * t for rn, t in zip(Rn, taus)]
        lx = [_m.log(t) for t in taus]; ly = [_m.log(max(v,1e-30)) for v in R]
        n=len(lx); mx=sum(lx)/n; my=sum(ly)/n
        sl = sum((a-mx)*(b-my) for a,b in zip(lx,ly))/sum((a-mx)**2 for a in lx)
        print(f"  {name:20s}  R/tau={['%.2e'%v for v in Rn]}  exponent(R~tau^?)={sl:.2f}")
    print("  expect: affine ~0 (exact), smooth exponent~2 (R/tau->0), kink exponent~1 (R/tau->c>0)")

# =====================================================================
if __name__ == "__main__":
    print(f"device = {dev}, dtype = {DT}")
    testbed_A()
    testbed_B_line_metric()
    testbed_C_circle()
    testbed_D_residual()
    print("\n[P0 done] Dán toàn bộ output này lại; tao chỉnh caption/số trong main.tex theo số thật.")
