import torch
dev = 'cuda'

# ---------- ham test (2-kink ReLU) ----------
def f(x):
    x1, x2 = x[..., 0], x[..., 1]
    return torch.relu(x1 - x2) * x2 + torch.relu(x1 + x2 - 1) - 0.5 * x1

# closed-form gradient (ground truth) de doi chieu
def grad_true(x):
    x1, x2 = x[..., 0], x[..., 1]
    g1 = (x1 > x2).float()
    g2 = (x1 + x2 > 1).float()
    d1 = g1 * x2 + g2 * 1.0 - 0.5          # df/dx1
    d2 = g1 * (x1 - 2*x2) + g2 * 1.0        # df/dx2
    return torch.stack([d1, d2], -1)

# ============================================================
# 1. LOESS local linear surrogate -> (w, residual, n_eff)
#    khong masking, khong baseline, chi data that lan can
# ============================================================
def loess(Y, X, x0, tau=0.05):             # x0: (M,2)
    K = torch.exp(-((X[None] - x0[:, None])**2).sum(-1) / (2 * tau**2)).sqrt()
    W, R, N = [], [], []
    for m in range(len(x0)):
        A = torch.cat([torch.ones(len(X), 1, device=dev), X - x0[m]], 1) * K[m][:, None]
        beta = torch.linalg.lstsq(A, (Y * K[m])[:, None]).solution.squeeze()
        W.append(beta[1:])
        R.append((A @ beta - Y * K[m]).pow(2).sum() / K[m].pow(2).sum())
        N.append(K[m].pow(2).sum()**2 / K[m].pow(4).sum())   # Kish n_eff
    return torch.stack(W), torch.stack(R).sqrt(), torch.stack(N)

# ============================================================
# 2. Integrated Gradients (Aumann-Shapley) tren duong thang baseline->x
#    df/dx1 tich phan doc path -> DINH OOD khi path cat kink
# ============================================================
def integrated_gradients(x, baseline, steps=2000):
    ts = torch.linspace(0, 1, steps, device=dev).view(-1, 1)
    path = baseline + ts * (x - baseline)      # (steps, 2)
    g = grad_true(path)                        # dung grad giai tich cho sach
    ig = (x - baseline) * g.mean(0)            # trapezoid ~ mean
    return ig                                  # (2,) -> phi_1, phi_2

# ============================================================
# 3. Exact Shapley 2-bien (masking = thay bang gia tri baseline)
#    v(S) = f(x_S, baseline_{~S}) -> DINH OOD: (x_S, baseline) la diem bia
# ============================================================
def shapley_2d(x, baseline):
    b = baseline
    v_empty = f(b)
    v_1 = f(torch.stack([x[0], b[1]]))         # chi bat x1
    v_2 = f(torch.stack([b[0], x[1]]))         # chi bat x2
    v_full = f(x)
    # Shapley 2 bien: phi_i = 1/2[(v_i - v_empty) + (v_full - v_{~i})]
    phi_1 = 0.5 * ((v_1 - v_empty) + (v_full - v_2))
    phi_2 = 0.5 * ((v_2 - v_empty) + (v_full - v_1))
    return torch.stack([phi_1, phi_2])

# ============================================================
# DEMO: case study diem C (0.2, 1.5), baseline (0,0)
# ============================================================
X = 4 * torch.rand(400_000, 2, device=dev) - 1     # data that [-1,3]^2
pt = torch.tensor([0.2, 1.5], device=dev)
bl = torch.tensor([0.0, 0.0], device=dev)

gt   = grad_true(pt)                               # (+0.5, +1.0) tai C
w, r, n = loess(f(X), X, pt.unsqueeze(0))
ig   = integrated_gradients(pt, bl)
shap = shapley_2d(pt, bl)

print(f"Ground truth  df/dx :  {gt.tolist()}")
print(f"LOESS      w        :  {w[0].tolist()}   res={r.item():.4f}  n_eff={n.item():.0f}")
print(f"IG         phi      :  {ig.tolist()}")
print(f"SHAP       phi      :  {shap.tolist()}")

# --- detector regime-mixing mien phi (Lemma 3.5): gap |IG - SHAP| ---
gap = (ig - shap).abs()
print(f"\n|IG - SHAP|         :  {gap.tolist()}   (!=0 => path cat kink)")

# --- doi chung: baseline (1.5,0.5) cung vung A -> path KHONG cat kink ---
# theo lemma: IG == SHAP chinh xac
bl2 = torch.tensor([1.5, 0.5], device=dev)
pt2 = torch.tensor([2.0, 0.8], device=dev)         # cung o vung A
ig2   = integrated_gradients(pt2, bl2)
shap2 = shapley_2d(pt2, bl2)
print(f"\n[control, path trong 1 vung A]")
print(f"IG   : {ig2.tolist()}")
print(f"SHAP : {shap2.tolist()}")
print(f"gap  : {(ig2-shap2).abs().tolist()}  (~0 => xac nhan lemma 3.5)")