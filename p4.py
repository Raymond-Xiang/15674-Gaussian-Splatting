"""P4: densified fit (256 -> 1024 Gaussians) vs. plain fixed-count fit (1024 Gaussians).

Usage: python p4.py [image_name]      (default: coffee, read from data/2d/<image_name>.png)
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

from gaussian_2d import get_device, covariance_2d, render, densify, save_img


def fit(path, N0, device, budget=None, steps=2000, densify_every=200, top_frac=0.25, seed=0):
    """Fit one image. budget=None gives a plain fixed-count fit with N0 Gaussians;
    otherwise start from N0 and densify toward `budget`."""
    torch.manual_seed(seed)
    target = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).float().div(255).to(device)
    H, W = target.shape[:2]

    mu = (torch.rand(N0, 2, device=device) * torch.tensor([W, H], device=device)).requires_grad_()
    log_s = torch.log(0.02 * max(H, W) * torch.ones(N0, 2, device=device)).requires_grad_()
    theta = torch.zeros(N0, device=device, requires_grad=True)
    color = torch.zeros(N0, 3, device=device, requires_grad=True)
    op_raw = torch.full((N0,), -2.0, device=device, requires_grad=True)
    # one parameter per group, in the order densify() expects
    opt = torch.optim.Adam([
        {"params": [mu], "lr": 0.2},
        {"params": [log_s], "lr": 1e-2},
        {"params": [theta], "lr": 1e-2},
        {"params": [color], "lr": 1e-2},
        {"params": [op_raw], "lr": 1e-2},
    ])
    order = torch.arange(N0, device=device)
    g_acc = torch.zeros(N0, device=device)
    n_acc = 0

    for step in range(steps):
        Sigma = covariance_2d(log_s.exp(), theta)
        img = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), order, H, W)
        loss = ((img - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        g_acc += mu.grad.norm(dim=-1)          # position-gradient magnitude, accumulated between passes
        n_acc += 1
        opt.step()

        # densify every `densify_every` steps, and stop for the last 30% of training
        if budget and (step + 1) % densify_every == 0 and step < 0.7 * steps:
            mu, log_s, theta, color, op_raw = densify(
                mu, log_s, theta, color, op_raw, opt, g_acc / n_acc, budget, W, top_frac=top_frac)
            N = mu.shape[0]
            order = torch.arange(N, device=device)
            g_acc = torch.zeros(N, device=device)
            n_acc = 0
        if (step + 1) % 200 == 0:
            print(f"  step {step + 1}  N {mu.shape[0]}  psnr {(-10 * torch.log10(loss)).item():.2f}", flush=True)

    with torch.no_grad():
        Sigma = covariance_2d(log_s.exp(), theta)
        img = render(mu, Sigma, color.sigmoid(), op_raw.sigmoid(), order, H, W).clamp(0, 1)
        psnr = (-10 * torch.log10(((img - target) ** 2).mean())).item()
    return psnr, img, mu.shape[0]


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "coffee"
    device = get_device()
    os.makedirs("results", exist_ok=True)

    runs = {"densified": dict(N0=256, budget=1024),      # grow from 256 to 1024
            "fixed": dict(N0=1024, budget=None)}         # plain fit at the same final count
    for tag, kw in runs.items():
        print(f"== {name}  {tag}", flush=True)
        psnr, img, n = fit(f"data/2d/{name}.png", device=device, **kw)
        save_img(img, f"results/p4_{name}_{tag}.png")
        print(f"{tag}: N = {n}  PSNR {psnr:.2f} dB", flush=True)
