#!/usr/bin/env python3
# =====================================================================
#  run_p1.py  --  P1 automatic geometry routing.
#
#  Chốt thiết kế (theo review):
#   * q ĐƯỢC CHỌN BỞI MASS DIMENSION  N(r) ~ r^m,  KHÔNG bởi eigengap.
#   * V bị abstain BỞI FLATNESS  eps_q(r) = sum_{j>q} lam_j / sum lam_j
#     plateau ở c>0 (không phải bởi eigenvector drift, không phải bởi
#     "instability" — V lý tưởng self-similar nên error có thể rất ổn định).
#   * drift D_q chỉ là gate phụ (bắt asymmetry/mixture), khong chung minh smooth.
#   * geometry router KHÔNG nhìn Y=f(X). Tách hẳn khỏi model regularity.
#
#  Torch. RUN: python run_p1.py   (CPU hoặc GPU, tự bắt)
#  Không ghi file. In certificate cho 5 support + 1 negative control.
# =====================================================================
import math
import torch

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
DT = torch.float64
torch.set_default_dtype(DT)

# ---------------------------------------------------------------------
#  Hyper-params của router (pre-registered; đổi ở đây nếu cần)
# ---------------------------------------------------------------------
RADII      = [0.40, 0.28, 0.20, 0.14, 0.10, 0.07, 0.05]   # giảm dần
N_MIN      = 200        # min điểm trong ball để một scale hợp lệ
DELTA_M    = 0.25       # |m_hat - q| phải < cái này
FLAT_TOL   = 0.02       # eps_q coi là ~0 nếu dưới ngưỡng này (noise floor)
FLAT_DROP  = 0.5        # eps_q phải giảm ít nhất hệ số này qua band -> "->0"
DRIFT_TOL  = 0.15       # principal-subspace drift Frobenius
GAP_MIN    = 3.0        # eigengap tối thiểu để ORIENT basis (khong infer dim)

# ---------------------------------------------------------------------
#  Đo trong ball compact bán kính r (geometry chỉ dùng X, không dùng Y)
# ---------------------------------------------------------------------
def ball_stats(X, x0, r):
    diff = X - x0
    r2 = (diff**2).sum(-1)
    m = r2 <= r*r
    n = int(m.sum().item())
    if n < 2:
        return dict(n=n, cov=None, evals=None, evecs=None)
    P = diff[m]
    mu = P.mean(0)
    Pc = P - mu
    cov = (Pc.T @ Pc) / P.shape[0]
    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.flip(0).clamp_min(0)         # descending, >=0
    evecs = evecs.flip(1)
    return dict(n=n, cov=cov, evals=evals, evecs=evecs)

def mass_dimension(X, x0, radii):
    """m_hat = slope(log N(r), log r) trên các scale đủ sample."""
    logr, logN, ns = [], [], []
    for r in radii:
        diff = X - x0
        n = int(((diff**2).sum(-1) <= r*r).sum().item())
        ns.append(n)
        if n >= N_MIN:
            logr.append(math.log(r)); logN.append(math.log(n))
    if len(logr) < 2:
        return None, ns
    lx = torch.tensor(logr); ly = torch.tensor(logN)
    mx, my = lx.mean(), ly.mean()
    slope = ((lx-mx)*(ly-my)).sum() / ((lx-mx)**2).sum()
    return float(slope), ns

def flatness(stats, q):
    """eps_q = sum_{j>q} lam_j / sum lam_j  (normalized reconstruction error)."""
    ev = stats['evals']
    if ev is None or ev.sum() <= 0:
        return None
    if q >= ev.numel():
        return 0.0
    return float(ev[q:].sum() / ev.sum())

def subspace_drift(sA, sB, q):
    """||U U^T - V V^T||_F giữa hai scale, cho q-plane hàng đầu."""
    if sA['evecs'] is None or sB['evecs'] is None:
        return None
    U = sA['evecs'][:, :q]; V = sB['evecs'][:, :q]
    return float(torch.linalg.norm(U@U.T - V@V.T))

def eigengap(stats, q):
    ev = stats['evals']
    if ev is None or q >= ev.numel():
        return float('inf')
    return float(ev[q-1] / ev[q].clamp_min(1e-30))

# ---------------------------------------------------------------------
#  GEOMETRY ROUTER  (chỉ nhìn X)
# ---------------------------------------------------------------------
def geometry_route(X, x0, radii=RADII, d=2):
    x0 = x0.reshape(-1)
    # --- 1. intrinsic dimension từ mass scaling ---
    m_hat, ns = mass_dimension(X, x0, radii)
    if m_hat is None:
        return dict(status='abstain', q=None, reason='insufficient-sample',
                    m_hat=None, counts=ns)
    q = int(round(m_hat))
    q = max(1, min(d, q))
    dim_ok = abs(m_hat - q) < DELTA_M

    # --- gom stats theo scale ---
    stats = [ball_stats(X, x0, r) for r in radii]
    valid = [i for i, s in enumerate(stats) if s['n'] >= N_MIN]

    # --- 2. flatness curve eps_q(r) trên các scale hợp lệ ---
    eps = [(radii[i], flatness(stats[i], q)) for i in valid]
    eps_vals = [e for _, e in eps if e is not None]

    # --- 3. drift giữa các scale liên tiếp hợp lệ ---
    drifts = []
    for a, b in zip(valid[:-1], valid[1:]):
        drifts.append((radii[a], radii[b], subspace_drift(stats[a], stats[b], q)))

    # --- eigengap (chỉ để orient basis khi q<d) ---
    gaps = [(radii[i], eigengap(stats[i], q)) for i in valid]

    # --- QUYẾT ĐỊNH ---
    # full-dim: q==d thì không cần chart, flatness vô nghĩa (eps_d=0 luôn)
    if q == d and dim_ok:
        return dict(status='ambient', q=d, reason='full-dimensional',
                    m_hat=m_hat, counts=ns, flatness=eps, drift=drifts, gaps=gaps,
                    band=[radii[i] for i in valid])

    # q<d: cần một q-plane THẬT SỰ tồn tại ở fine scale.
    # tiêu chí: tồn tại contiguous fine-scale suffix mà eps_q -> 0
    #           (giảm dần và chạm dưới noise floor), drift nhỏ, gap đủ orient.
    def suffix_pass():
        # duyệt từ suffix dài nhất (fine scales) trở lên
        for start in range(len(valid)):
            idxs = valid[start:]
            if len(idxs) < 2:
                continue
            evs = [flatness(stats[i], q) for i in idxs]
            if any(e is None for e in evs):
                continue
            # eps phải đi xuống và chạm noise floor ở scale mịn nhất
            goes_to_zero = (evs[-1] < FLAT_TOL) or (evs[-1] < FLAT_DROP*evs[0])
            drift_ok = all(
                (subspace_drift(stats[a], stats[b], q) or 9) < DRIFT_TOL
                for a, b in zip(idxs[:-1], idxs[1:]))
            gap_ok = all(eigengap(stats[i], q) > GAP_MIN for i in idxs) if q < d else True
            if goes_to_zero and drift_ok and gap_ok:
                return [radii[i] for i in idxs]
        return None

    band = suffix_pass()
    if not dim_ok:
        return dict(status='abstain', q=None, reason=f'dimension-unstable(m={m_hat:.2f})',
                    m_hat=m_hat, counts=ns, flatness=eps, drift=drifts, gaps=gaps)
    if band is None:
        # dimension ok (q=1) nhưng KHÔNG có q-plane phẳng ở fine scale -> V apex
        return dict(status='abstain', q=q, reason='not-locally-q-flat (eps_q plateau>0)',
                    m_hat=m_hat, counts=ns, flatness=eps, drift=drifts, gaps=gaps,
                    band=None)
    return dict(status='tangent', q=q, reason='q-flat-stable', m_hat=m_hat,
                counts=ns, flatness=eps, drift=drifts, gaps=gaps, band=band)

# ---------------------------------------------------------------------
#  Supports để test (chỉ X; router không thấy f)
# ---------------------------------------------------------------------
def gen_plane(n=400_000):
    return (4*torch.rand(n, 2, device=dev) - 1).to(DT)

def gen_line(n=400_000):
    t = (4*torch.rand(n, 1, device=dev) - 1).to(DT)
    return torch.cat([t, t], 1)

def gen_circle(n=400_000):
    th = (2*math.pi*torch.rand(n, device=dev)).to(DT)
    return torch.stack([th.cos(), th.sin()], 1)

def gen_V(n=400_000, a=0.25):
    # narrow symmetric V: (t, a|t|). a nhỏ -> eigengap ĐẸP, eigenvector STABLE.
    # nếu router vẫn abstain thì đó là nhờ flatness, không nhờ spectrum.
    t = (2*torch.rand(n, device=dev) - 1).to(DT)
    return torch.stack([t, a*t.abs()], 1)

def gen_rounded_V(n=400_000, a=0.25, eta=0.05):
    # boundary case: r<<eta smooth (eps_1->0), r>>eta V-like (eps_1 plateau)
    t = (2*torch.rand(n, device=dev) - 1).to(DT)
    y = a*(torch.sqrt(t*t + eta*eta) - eta)
    return torch.stack([t, y], 1)

# ---------------------------------------------------------------------
def report(name, X, x0, expect):
    r = geometry_route(X, x0)
    print(f"\n--- {name}  (expect: {expect}) ---")
    print(f"  status = {r['status']:8s}  q = {r['q']}   reason = {r['reason']}")
    if r.get('m_hat') is not None:
        print(f"  mass-dim m_hat = {r['m_hat']:.3f}")
    if r.get('flatness'):
        fl = "  ".join(f"r={rr:.2f}:{ee:.3f}" for rr, ee in r['flatness'] if ee is not None)
        print(f"  flatness eps_q(r):  {fl}")
    if r.get('band'):
        print(f"  selected fine-scale band = {r['band']}")
    if r.get('gaps'):
        gp = "  ".join(f"r={rr:.2f}:{gg:.1f}" for rr, gg in r['gaps'])
        print(f"  eigengap(r) [orient only]: {gp}")

if __name__ == "__main__":
    print(f"device = {dev}, dtype = {DT}")
    print("GEOMETRY ROUTER (chỉ nhìn X; f không được nhìn tới)")
    o = torch.tensor([0.0, 0.0], device=dev, dtype=DT)
    report("Plane R^2",           gen_plane(),     o, "ambient(q=2)")
    report("Line x1=x2",          gen_line(),      o, "tangent(q=1)")
    report("Circle S^1",  gen_circle(), torch.tensor([1.,0.],device=dev,dtype=DT),
           "tangent(q=1)")
    report("Narrow V a=0.25 (NEG CONTROL)", gen_V(a=0.25), o,
           "abstain(not-q-flat)  <-- eigengap ĐẸP nhưng eps_1 plateau")
    report("Rounded V a=0.25 eta=0.05", gen_rounded_V(), o,
           "tangent(q=1) nếu đủ sample dưới eta; else resolution-limited")
    print("\n[P1 done] Dán output. Điểm mấu chốt cần thấy:")
    print("  * Narrow V: mass-dim ~1, eigengap lớn, NHƯNG eps_1 KHÔNG ->0  => abstain")
    print("  * Circle:   mass-dim ~1, eps_1 ~ r^2 ->0                      => tangent(q=1)")
    print("  * Plane:    mass-dim ~2                                       => ambient(q=2)")