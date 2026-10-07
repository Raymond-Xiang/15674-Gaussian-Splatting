"""P7 / P8: train the plain fixed-count 3D fit and the densified 3D fit, then evaluate both.

Usage: python p7_p8.py
Writes renders, PSNR and trained parameters to results/ (p8_params.pt is used by the P9 orbit
render in gaussian_3d.py).
"""
import os

import torch

from gaussian_3d import get_device, load_scene, train_3d, evaluate

if __name__ == "__main__":
    device = get_device()
    os.makedirs("results", exist_ok=True)
    K, H, W, train, val = load_scene("data/spheres", device)
    steps, budget = 3000, 2000

    runs = {"p7": dict(N0=budget, budget=None),                      # P7: fixed cloud of 2000 Gaussians
            "p8": dict(N0=200, budget=budget, init_scale=0.15)}      # P8: start from 200, densify to 2000
    for tag, kw in runs.items():
        print("==", tag, flush=True)
        P = train_3d(train, K, H, W, device, steps=steps, **kw)
        tr = evaluate(P, train, K, H, W, prefix=f"results/{tag}_train")
        va = evaluate(P, val, K, H, W, prefix=f"results/{tag}_val")
        print(f"{tag}: N = {P[0].shape[0]}  train PSNR {tr:.2f}  val PSNR {va:.2f}", flush=True)
        torch.save([p.detach().cpu() for p in P], f"results/{tag}_params.pt")
