import torch
torch.set_printoptions(precision=4, sci_mode=False)
dev = 'cuda'

def f(x):
    x1, x2 = x[..., 0], x[..., 1]
    return torch.relu(x1 - x2) * x2 + torch.relu(x1 + x2 - 1) - 0.5 * x1

X = 4 * torch.rand(200_000, 2, device=dev) - 1        # "data that" tren [-1,3]^2
Y = f(X)
X = 4 * torch.rand(400_000, 2, device=dev) - 1
Z = 5 * f(X)                                   # logit, scale 5 cho bao hoa ro
P = torch.sigmoid(Z)

def loess_w(Y, x0, tau=0.15):                  # x0: (M,2)
    K = torch.exp(-((X[None] - x0[:, None])**2).sum(-1) / (2*tau**2)).sqrt()
    w = []
    for m in range(len(x0)):
        A = torch.cat([torch.ones(len(X),1,device=dev), X - x0[m]], 1) * K[m][:,None]
        w.append(torch.linalg.lstsq(A, (Y*K[m])[:,None]).solution.squeeze()[1:])
    return torch.stack(w)

pts = torch.tensor([[2.0, 0.5], [2.5, 1.0], [1.2, 0.8]], device=dev)  # sau trong vung A, p ~ 1
print('w tren p     :', loess_w(P, pts).cpu())   # ky vong: chet ~0 het
print('w tren logit :', loess_w(Z, pts).cpu())   # ky vong: 5*(x2+0.5, x1-2x2+1), song nguyen