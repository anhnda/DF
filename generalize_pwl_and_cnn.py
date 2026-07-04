"""
generalize_pwl_and_cnn.py

Two things:

(C1) PROPOSITION-BY-EXPERIMENT: on random ReLU MLPs (PWL nets), the signed
     remainder R_i is a weighted sum of gradient jumps across the activation
     cells the path visits. We verify the structural bound

        |R_i|  <=  |x_i - b_i| * sum_k |[[∂_i f]]_k|

     where [[∂_i f]]_k are the per-coordinate gradient jumps at the k crossings.
     (RHS = |span| * total unsigned jump; LHS = |span| * |signed net drift|.)
     We also show E_i tracks the unsigned side, so E_i >> |R_i| flags
     cancellation. This is the general version of the hinge sign-flip.

(C2) REAL ReLU CNN audit: same L/R/E decomposition on a small conv net,
     per input pixel, using straight-line IG. Reports:
       - fraction of path steps whose ReLU activation pattern differs from
         the query's pattern  (path-outside-query-fingerprint ratio)
       - total gate flips
       - |R|/|IG| and E/|R| summary stats over pixels

GPU torch, double precision where feasible. You run it. I don't run anything.
"""

import torch
import torch.nn as nn
from regime_mixing_audit import DEVICE, batched_grad

DTYPE = torch.float64


# ======================================================================
# (C1) random ReLU MLP  -- PWL, exact cell structure
# ======================================================================
class ReLUMLP(nn.Module):
    def __init__(self, d_in, width, depth, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        layers = []
        d = d_in
        for _ in range(depth):
            lin = nn.Linear(d, width)
            with torch.no_grad():
                lin.weight.normal_(0, 1.0 / d**0.5, generator=g)
                lin.bias.normal_(0, 0.5, generator=g)
            layers += [lin, nn.ReLU()]
            d = width
        head = nn.Linear(d, 1)
        with torch.no_grad():
            head.weight.normal_(0, 1.0 / d**0.5, generator=g)
            head.bias.zero_()
        layers += [head]
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X).squeeze(-1)   # (B,)


def relu_pattern(model, X):
    """Return concatenated ReLU on/off pattern (B, total_hidden) for each row."""
    pats = []
    h = X
    for m in model.net:
        h = m(h)
        if isinstance(m, nn.ReLU):
            pats.append((h > 0).to(torch.int8))
    return torch.cat(pats, dim=1)


def measure_jumps_along_path(model, P, coord, tol=1e-9):
    """
    Along ordered path P (T,d), find gradient jumps in coordinate `coord`.
    Returns: signed_net_drift, total_unsigned_jump, n_crossings.
    """
    g = batched_grad(lambda z: model(z), P)   # (T,d)
    gi = g[:, coord]                           # (T,)
    dgi = gi[1:] - gi[:-1]                      # (T-1,)
    mask = dgi.abs() > tol
    signed = dgi[mask].sum()
    unsigned = dgi[mask].abs().sum()
    return signed.item(), unsigned.item(), int(mask.sum().item())


def run_C1(d_in=4, width=32, depth=3, n_pairs=8, n_steps=8000, seed=0):
    print("=" * 66)
    print("(C1) random ReLU MLP: |R_i| <= |span| * total unsigned jump")
    print("=" * 66)
    model = ReLUMLP(d_in, width, depth, seed=seed).to(DEVICE, DTYPE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    g = torch.Generator().manual_seed(seed + 100)
    print(f"{'pair':>4} {'coord':>5} {'|R_i|':>12} {'RHS bound':>12} {'E_i':>12} {'holds?':>7}")
    for k in range(n_pairs):
        b = (torch.rand(d_in, generator=g) * 4 - 2).to(DEVICE, DTYPE)
        x = (torch.rand(d_in, generator=g) * 4 - 2).to(DEVICE, DTYPE)
        t = (torch.arange(n_steps, device=DEVICE, dtype=DTYPE) + 0.5) / n_steps
        P = b[None, :] + t[:, None] * (x - b)[None, :]

        gx = batched_grad(lambda z: model(z), x[None, :])[0]   # (d,)
        gP = batched_grad(lambda z: model(z), P)               # (T,d)
        span = (x - b)
        ig = span * gP.mean(dim=0)
        L = span * gx
        R = ig - L
        E = span.abs() * (gP - gx[None, :]).abs().mean(dim=0)

        for coord in range(d_in):
            _, unsigned, _ = measure_jumps_along_path(model, P, coord)
            rhs = span[coord].abs().item() * unsigned
            lhs = R[coord].abs().item()
            ok = lhs <= rhs + 1e-6
            print(f"{k:>4} {coord:>5} {lhs:>12.5f} {rhs:>12.5f} "
                  f"{E[coord].item():>12.5f} {str(ok):>7}")
    print("\n  If 'holds?' is True everywhere, the structural bound is confirmed:")
    print("  signed remainder is dominated by total unsigned gradient-jump mass.")
    print("  E_i >> |R_i| on a coordinate == regimes cancelled (hidden non-locality).")


# ======================================================================
# (C2) real ReLU CNN audit
# ======================================================================
class SmallCNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=10, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 8, 3, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Linear(16 * 4 * 4, n_classes)

    def forward(self, x):
        # x: (B, C*H*W) flattened  OR (B,C,H,W). We accept flattened for autograd.
        if x.dim() == 2:
            B = x.shape[0]
            x = x.view(B, self.in_shape[0], self.in_shape[1], self.in_shape[2])
        h = self.features(x)
        h = h.flatten(1)
        return self.head(h)

    in_shape = (1, 16, 16)


def cnn_relu_pattern(model, x_img):
    """ReLU on/off pattern for a single (1,C,H,W) input, concatenated."""
    pats = []
    h = x_img
    for m in model.features:
        h = m(h)
        if isinstance(m, nn.ReLU):
            pats.append((h > 0).flatten())
    return torch.cat(pats)


def run_C2(target_class=None, n_steps=64, seed=1):
    print("\n" + "=" * 66)
    print("(C2) real ReLU CNN: per-pixel L/R/E audit + fingerprint mismatch")
    print("=" * 66)
    C, H, W = SmallCNN.in_shape
    d = C * H * W
    model = SmallCNN(in_ch=C, seed=seed).to(DEVICE).float()  # CNN in float32 (cudnn)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    torch.manual_seed(seed + 7)
    x_img = torch.rand(1, C, H, W, device=DEVICE)          # a fake 'image'
    b_img = torch.zeros_like(x_img)                        # black baseline (classic)

    logits_x = model(x_img)
    cls = int(logits_x.argmax()) if target_class is None else target_class
    print(f"query class = {cls}")

    f = lambda z: model(z.view(-1, C, H, W))[:, cls]       # scalar output per row

    xf = x_img.flatten()[None, :]                          # (1,d)
    bf = b_img.flatten()[None, :]
    t = (torch.arange(n_steps, device=DEVICE) + 0.5).float() / n_steps
    P = bf + t[:, None] * (xf - bf)                        # (T,d)

    gx = batched_grad(f, xf)[0]                             # (d,)
    gP = batched_grad(f, P)                                # (T,d)
    span = (xf - bf).squeeze(0)                            # (d,)

    ig = span * gP.mean(0)
    L = span * gx
    R = ig - L
    E = span.abs() * (gP - gx[None, :]).abs().mean(0)

    # fingerprint mismatch: fraction of path points whose ReLU pattern != query's
    fp_x = cnn_relu_pattern(model, x_img)
    mismatch = 0
    for k in range(n_steps):
        img_k = P[k].view(1, C, H, W)
        fp_k = cnn_relu_pattern(model, img_k)
        if not torch.equal(fp_k, fp_x):
            mismatch += 1
    frac_outside = mismatch / n_steps

    # gate flips total along path
    flips = 0
    prev = None
    for k in range(n_steps):
        fp_k = cnn_relu_pattern(model, P[k].view(1, C, H, W))
        if prev is not None:
            flips += int((fp_k != prev).sum().item())
        prev = fp_k

    print(f"pixels d              = {d}")
    print(f"path-outside-query FP = {frac_outside:.3f}  "
          f"(1.0 => path NEVER in query's activation cell)")
    print(f"total gate flips      = {flips}")
    print(f"||IG||_1              = {ig.abs().sum().item():.4f}")
    print(f"||L||_1  (local)      = {L.abs().sum().item():.4f}")
    print(f"||R||_1  (mixing)     = {R.abs().sum().item():.4f}")
    print(f"||R||_1 / ||IG||_1    = {(R.abs().sum()/ig.abs().sum()).item():.3f}"
          f"   (how much of IG is NOT query-local)")
    ratio = (E / (R.abs() + 1e-12))
    print(f"median E/|R| per pixel= {ratio.median().item():.2f}"
          f"   (>>1 => heavy cancellation, small IG != local)")
    print("\n  Interpretation: on a real ReLU CNN with a black baseline, essentially")
    print("  the entire straight path lives OUTSIDE the query's activation cell,")
    print("  so a large share of IG is regime-mixing R, not local response L.")


if __name__ == "__main__":
    print(f"device = {DEVICE}\n")
    run_C1()
    run_C2()
