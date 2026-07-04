import torch
torch.set_printoptions(precision=4, sci_mode=False)
dev = 'cuda'

def f(x):
    x1, x2 = x[..., 0], x[..., 1]
    return torch.relu(x1 - x2) * x2 + torch.relu(x1 + x2 - 1) - 0.5 * x1

X = 4 * torch.rand(200_000, 2, device=dev) - 1        # "data that" tren [-1,3]^2
Y = f(X)

def local_w(x0, tau=0.15):                            # x0: (M,2) -> w: (M,2)
    d2 = ((X[None] - x0[:, None])**2).sum(-1)         # (M,N)
    K = torch.exp(-d2 / (2 * tau**2)).sqrt()
    out = []
    for m in range(len(x0)):
        A = torch.cat([torch.ones(len(X), 1, device=dev), X - x0[m]], 1) * K[m][:, None]
        out.append(torch.linalg.lstsq(A, (Y * K[m])[:, None]).solution.squeeze()[1:])
    return torch.stack(out)

g = torch.stack(torch.meshgrid(torch.linspace(-.5, 2.5, 31, device=dev),
                               torch.linspace(-.5, 2.5, 31, device=dev), indexing='ij'), -1).reshape(-1, 2)
W = local_w(g).reshape(31, 31, 2)
dw1_dx2 = (W[:, 1:, 0] - W[:, :-1, 0]) / 0.1          # ky vong ~ 1[x1>x2], nhoe quanh kink