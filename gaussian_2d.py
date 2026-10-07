import torch
from PIL import Image
import numpy as np
import math
import os, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def covariance_2d(scale, theta):
    c, s = torch.cos(theta), torch.sin(theta)        # both (N,)
    row0 = torch.stack([c, -s], dim=-1)              # (N, 2)
    row1 = torch.stack([s,  c], dim=-1)              # (N, 2)
    R = torch.stack([row0, row1], dim=-2)            # (N, 2, 2)
    S = torch.diag_embed(scale)                      # (N, 2, 2)
    M = R @ S
    return M @ M.transpose(-1, -2)                   # (N, 2, 2)

def pixel_grid(H, W, device=None):
    ys, xs = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing='ij')
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2).float()   # (P, 2)


# def gaussian_weight(xy, mu, Sigma):
#     #xy is a grid of pixel coordinates

#     d = xy[:, None, :] - mu[None, :, :]                  # (P, N, 2): offsets from each pixel to the gaussian mean
#     Sigma_inv = torch.linalg.inv(Sigma)                  # (N, 2, 2): inverse of covariance
#     maha = torch.einsum('pni,nij,pnj->pn', d, Sigma_inv, d)   # (P, N) i.e. dᵀ Σ⁻¹ d, how "far" away is each pixel??
#     return torch.exp(-0.5 * maha)                        # (P, N): w(x) = e^(-0.5* (x-u)' * Sigma^-1 * (x-u))

# maybe this  runs faster  ....
def gaussian_weight(xy, mu, Sigma):
    a, b, c = Sigma[:, 0, 0], Sigma[:, 0, 1], Sigma[:, 1, 1]     # 各 (N,)
    det = a * c - b * b
    dx = xy[:, 0:1] - mu[None, :, 0]                             # (P, N)
    dy = xy[:, 1:2] - mu[None, :, 1]                             # (P, N)
    maha = (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / det   # dᵀ Σ⁻¹ d
    return torch.exp(-0.5 * maha)

# def render(mu, Sigma, color, opacity, order, H, W):
#     # color: (N, 3),  opacity: (N,) in [0, 1],  order: indices sorted front -> back
#     xy = pixel_grid(H, W)                     # (H*W, 2)
#     w  = gaussian_weight(xy, mu, Sigma)       # (P, N)  from P1
#     alpha = opacity[None, :] * w              # (P, N)
#     C = torch.zeros(H * W, 3)
#     T = torch.ones(H * W)
#     for i in order:                           # front to back
#         a = alpha[:, i]                               # (P,), alpha for each pixel for current gaussian
#         C = C + (a * T)[:, None] * color[i][None, :]  # (P,1)*(1,3) -> (P,3), updated color for each pixel, accumulated
#         T = T * (1 - a)                               # (P,), updated transmittance

#     return C.reshape(H, W, 3)

# a version that runs faster...
def render(mu, Sigma, color, opacity, order, H, W):
    xy = pixel_grid(H, W, device=mu.device)
    w  = gaussian_weight(xy, mu, Sigma)
    alpha = (opacity[None, :] * w)[:, order]                  # (P, N), in front-to-back order
    T = torch.cumprod(1 - alpha, dim=1)                       # accumulate by multiplying
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], 1)  # need an extra right shift by 1 place to ensure each column's transmiittance does not include itself
    return ((alpha * T) @ color[order]).reshape(H, W, 3)      # (P,N)@(N,3) -> (P,3)

def swap_param(opt, group, keep, extra):
    """keep: which ones to keep (bool mask); extra: new lines added"""
    old = group["params"][0]
    new = torch.cat([old.data[keep], extra]).requires_grad_()
    st = opt.state.pop(old, None)
    if st:
        for k in ("exp_avg", "exp_avg_sq"):
            st[k] = torch.cat([st[k][keep], torch.zeros_like(extra)])
        opt.state[new] = st
    group["params"][0] = new
    return new

@torch.no_grad()
def densify(mu, log_s, theta, color, op_raw, opt, g, budget, W,
            size_threshold=0.02, split_scale=1.6, prune_opacity=0.005, top_frac=0.1):
    """compute 4 masks, 1 dense -> 2 prune, 3 split, 4 clone"""
    
    max_scale = log_s.exp().max(dim=1).values              # (N,) the longer axis for each gaussian
    prune = op_raw.sigmoid() < prune_opacity               # (N,)

    dense = (g > torch.quantile(g, 1 - top_frac)) & ~prune

    room = budget - int((~prune).sum()) 
    if int(dense.sum()) > room:
        idx = torch.where(dense)[0]
        idx = idx[g[idx].argsort(descending=True)[:max(room, 0)]]
        dense = torch.zeros_like(dense)
        dense[idx] = True

    big   = max_scale > size_threshold * W
    split = dense & big
    clone = dense & ~big
    keep  = ~prune & ~split 
    
    #log 1.6
    lg = math.log(split_scale)

    # these are the new lines I am going to add
    theta_extra = torch.cat([theta[clone],  theta[split],       theta[split]])
    color_extra = torch.cat([color[clone],  color[split],       color[split]])
    op_extra    = torch.cat([op_raw[clone], op_raw[split],      op_raw[split]])
    log_s_extra = torch.cat([log_s[clone],  log_s[split] - lg,  log_s[split] - lg])
    
    s, th = log_s[split].exp(), theta[split]
    c, sn = th.cos(), th.sin()
    def child_mu():
        e = torch.randn_like(s) * s                              # (n, 2) 椭圆自身坐标系下的偏移
        off = torch.stack([c * e[:, 0] - sn * e[:, 1],
                           sn * e[:, 0] + c * e[:, 1]], dim=-1)  # 乘 R 转到图像坐标
        return mu[split] + off
    mu_extra = torch.cat([mu[clone], child_mu(), child_mu()])
    
    gs = opt.param_groups
    mu     = swap_param(opt, gs[0], keep, mu_extra)
    log_s  = swap_param(opt, gs[1], keep, log_s_extra)
    theta  = swap_param(opt, gs[2], keep, theta_extra)
    color  = swap_param(opt, gs[3], keep, color_extra)
    op_raw = swap_param(opt, gs[4], keep, op_extra)
    return mu, log_s, theta, color, op_raw


def save_img(t, path):
    arr = (t.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def fit(path, N, device, H=256, W=256, steps=2000, seed=0):
    torch.manual_seed(seed)
    # pil = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    pil = Image.open(path).convert("RGB")
    target = torch.from_numpy(np.array(pil)).float().div(255).to(device)

    # parameters (leaf tensors, requires_grad=True); a spread-out init, e.g.:
    # mu (N, 2)  spread across the image
    # log_s (N, 2)  small blobs, log space
    # theta (N,)    rotation
    # color (N, 3)  sigmoid -> 0.5 gray
    # op_raw (N,)    sigmoid -> ~0.12 opacity
    mu = (torch.rand(N, 2, device=device) * torch.tensor([W, H], device=device)).requires_grad_()
    log_s = torch.log(0.02 * max(H, W) * torch.ones(N, 2, device=device)).requires_grad_()
    theta = torch.zeros(N, device=device, requires_grad=True)
    color = torch.zeros(N, 3, device=device, requires_grad=True)
    op_raw = torch.full((N,), -2.0, device=device, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [mu], "lr": 0.2},
        {"params": [log_s, theta, color, op_raw], "lr": 1e-2},
    ])
    # 2D does not have actual depth value, so an arbitrary order is selected
    order = torch.arange(N, device=device)

    for step in range(steps):
        Sigma = covariance_2d(log_s.exp(), theta)
        img = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), order, H, W)
        loss = ((img - target) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step + 1}  psnr {(-10 * torch.log10(loss)).item():.2f}", flush=True)

    with torch.no_grad():                       # render with final parameters to get PSNR
        Sigma = covariance_2d(log_s.exp(), theta)
        img = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), order, H, W).clamp(0, 1)
        psnr = (-10 * torch.log10(((img - target) ** 2).mean())).item()
    return psnr, img, target



# Training script goes here

if __name__ == "__main__":
    device = get_device()
    os.makedirs("results", exist_ok=True)
    names = ["coffee", "cat", "astronaut"]
    counts = [256, 1024, 4096]
    results = {}

    for N in counts:
        for name in names:
            print(f"== {name}  N={N}", flush=True)
            t0 = time.time()
            try:
                psnr, img, target = fit(f"data/2d/{name}.png", N, device)
                results[f"{name}_{N}"] = psnr
                save_img(img, f"results/{name}_{N}.png")
                save_img(target, f"results/{name}_target.png")
                print(f"   PSNR {psnr:.2f} dB   ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"   FAILED: {e}", flush=True)
                torch.cuda.empty_cache()
            with open("results/p5_psnr.json", "w") as f:
                json.dump(results, f, indent=2)

    plt.figure(figsize=(5, 4))
    for name in names:
        pts = [(N, results[f"{name}_{N}"]) for N in counts if f"{name}_{N}" in results]
        plt.plot(*zip(*pts), marker="o", label=name)
    plt.xscale("log", base=2); plt.xticks(counts, counts)
    plt.xlabel("number of Gaussians N"); plt.ylabel("PSNR (dB)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/p5_psnr_vs_n.png", dpi=150)