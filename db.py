import torch
torch.set_printoptions(precision=4, sci_mode=False)
dev = 'cuda'

def f(x):
    x1, x2 = x[..., 0], x[..., 1]
    return torch.relu(x1 - x2) * x2 + torch.relu(x1 + x2 - 1) - 0.5 * x1

def grad(x):
    x = x.clone().requires_grad_(True)
    return torch.autograd.grad(f(x).sum(), x)[0]

def gxi(x, b):                       # gradient * (input - baseline)
    return grad(x) * (x - b)

def ig(x, b, steps=2048):            # Riemann midpoint IG
    t = (torch.arange(steps, device=dev) + 0.5) / steps
    path = b + t[:, None, None] * (x - b)              # (S,N,2)
    g = grad(path.reshape(-1, 2)).reshape(steps, -1, 2)
    return (x - b) * g.mean(0)

def shap2(x, b):                     # exact Shapley, 2 features
    xb = torch.stack([x[:, 0], b[:, 1]], 1)
    bx = torch.stack([b[:, 0], x[:, 1]], 1)
    fx, fb, fxb, fbx = f(x), f(b), f(xb), f(bx)
    p1 = 0.5 * ((fxb - fb) + (fx - fbx))
    p2 = 0.5 * ((fbx - fb) + (fx - fxb))
    return torch.stack([p1, p2], 1)

def cross(x):                        # d2f/dx1dx2, ground truth = 1[x1>x2]
    xr = x.clone().requires_grad_(True)
    g1 = torch.autograd.grad(f(xr).sum(), xr, create_graph=True)[0][:, 0]
    return torch.autograd.grad(g1.sum(), xr)[0][:, 1]

pts = torch.tensor([[-1.5, 0.2],    # D: dead zone, x2 phai = 0
                    [ 1.5, 1.0],    # A: bilinear + linear
                    [ 1.0, 0.2],    # B: g1=1, g2=0
                    [ 0.2, 1.5]],   # C: additive thuan
                   device=dev)
b0 = torch.zeros_like(pts)                                   # baseline goc
bA = torch.tensor([[1.5, 0.5]], device=dev).expand_as(pts)   # baseline nam trong vung A

for name, b in [('baseline (0,0)', b0), ('baseline (1.5,0.5)', bA)]:
    print(f'== {name}')
    print('GxI :\n', gxi(pts, b).cpu())
    print('IG  :\n', ig(pts, b).cpu())
    print('SHAP:\n', shap2(pts, b).cpu())

print('cross-deriv (ky vong 1[x1>x2] = [0,1,1,0]):', cross(pts).cpu())
print('grad tai dead point (ky vong [-0.5, 0]):', grad(pts[:1]).cpu())