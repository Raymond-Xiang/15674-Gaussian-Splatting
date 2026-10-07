from gaussian_2d import get_device, pixel_grid, gaussian_weight, swap_param, save_img
import torch
from PIL import Image
import numpy as np
import math
import os, json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# this should run faster and help 3d gaussian rendering
def composite(xy, mu, Sigma, color, opacity):
    """gaussians in front-to-back order, returns (P, 3)"""
    w = gaussian_weight(xy, mu, Sigma)                    # (P, n)
    alpha = (opacity[None, :] * w).clamp(max=0.99)
    l = torch.log1p(-alpha)                               # log(1 - α)
    T = torch.exp(torch.cumsum(l, dim=1) - l)
    return (alpha * T) @ color

def render_dense(mu, Sigma, color, opacity, order, H, W):
    xy = pixel_grid(H, W, device=mu.device)
    w  = gaussian_weight(xy, mu, Sigma)
    alpha = (opacity[None, :] * w)[:, order]                  # (P, N), in front-to-back order
    T = torch.cumprod(1 - alpha, dim=1)                       # accumulate by multiplying
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], 1)  # need an extra right shift by 1 place to ensure each column's transmiittance does not include itself
    return ((alpha * T) @ color[order]).reshape(H, W, 3)      # (P,N)@(N,3) -> (P,3)


def render(mu, Sigma, color, opacity, order, H, W, tile=32):
    mu, Sigma, color, opacity = mu[order], Sigma[order], color[order], opacity[order]
    with torch.no_grad():                                 # 3 sigma bounding box
        rx, ry = 3 * Sigma[:, 0, 0].sqrt(), 3 * Sigma[:, 1, 1].sqrt()
        x0, x1, y0, y1 = mu[:, 0] - rx, mu[:, 0] + rx, mu[:, 1] - ry, mu[:, 1] + ry
    rows = []
    for ty in range(0, H, tile):
        row = []
        for tx in range(0, W, tile):
            th, tw = min(tile, H - ty), min(tile, W - tx)
            m = (x1 > tx) & (x0 < tx + tw) & (y1 > ty) & (y0 < ty + th)   # 碰到这块的高斯
            xy = pixel_grid(th, tw, device=mu.device) + torch.tensor([tx, ty], device=mu.device)
            row.append(composite(xy, mu[m], Sigma[m], color[m], opacity[m]).reshape(th, tw, 3))
        rows.append(torch.cat(row, dim=1))
    return torch.cat(rows, dim=0)

def quaternion_to_rotation(q):
    q = q / q.norm(dim=-1, keepdim=True)                 # Normalize
    w, x, y, z = q.unbind(-1)

    #  Fill in the R matrix according to handout
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)], dim=-1)
    return R.reshape(-1, 3, 3)

def covariance_3d(scale, quat):
    R = quaternion_to_rotation(quat)
    M = R * scale[:, None, :]                            # equivalent to R @ diag(scale)

    #recall: Sigma = R * S * S' * R'
    return M @ M.transpose(-1, -2)                       # (N, 3, 3)

def project_gaussian(mu3, Sigma3, R_wc, t, K):
    mu_cam = mu3 @ R_wc.T + t                            # world to camera
    x, y = mu_cam[:, 0], mu_cam[:, 1]
    z = mu_cam[:, 2].clamp(min=0.1)                      # avoid division by 0
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mu2 = torch.stack([fx * x / z + cx, fy * y / z + cy], dim=-1)
    o = torch.zeros_like(z)
    J = torch.stack([fx / z, o, -fx * x / z**2,
                     o, fy / z, -fy * y / z**2], dim=-1).reshape(-1, 2, 3)
    Scam = R_wc @ Sigma3 @ R_wc.T                        # covariance transformed into camera coordinates
    Sig2 = J @ Scam @ J.transpose(-1, -2)                # projected onto screen (N, 2, 2)
    Sig2 = Sig2 + 0.3 * torch.eye(2, device=mu3.device)  # make it invertible
    return mu2, Sig2, mu_cam[:, 2]

def load_scene(root, device):
    meta = json.load(open(f"{root}/cameras.json"))
    K = torch.tensor(meta["K"], dtype=torch.float32, device=device)
    def load(frames):
        cams = []
        for f in frames:
            im = np.array(Image.open(f"{root}/{f['file']}").convert("RGB"))
            cams.append({
                "image": torch.from_numpy(im).float().div(255).to(device),
                "R": torch.tensor(f["R_wc"], dtype=torch.float32, device=device),
                "t": torch.tensor(f["t"], dtype=torch.float32, device=device)})
        return cams
    return K, meta["height"], meta["width"], load(meta["frames"]), load(meta["val_frames"])

def render_view(P, cam, K, H, W):
    mu3, log_s, quat, color, op_raw = P
    Sig3 = covariance_3d(log_s.exp(), quat)            # 3D scale + rotation
    mu2, Sig2, depth = project_gaussian(mu3, Sig3, cam["R"], cam["t"], K)
    opacity = op_raw.sigmoid() * (depth > 0.1)           # skip the gaussians behind camera
    return render(mu2, Sig2, color.sigmoid(), opacity, torch.argsort(depth), H, W) # front-to-back: nearest (smallest z_c) first

@torch.no_grad()
def evaluate(P, cams, K, H, W, prefix=None, n_save=3):
    psnrs = []
    for i, cam in enumerate(cams):
        img = render_view(P, cam, K, H, W).clamp(0, 1)
        psnrs.append((-10 * torch.log10(((img - cam["image"]) ** 2).mean())).item())
        if prefix and i < n_save:                        # left: render, right: ground truth
            save_img(torch.cat([img, cam["image"]], dim=1), f"{prefix}_{i:03d}.png")
    return sum(psnrs) / len(psnrs)


# (N, 3)  cloud in ~[-1.5, 1.5]^3
# (N, 3)  small 3D blobs
# (N, 4)  identity rotation (w, x, y, z)
# (N, 3)  sigmoid -> gray
# (N,)    sigmoid -> low opacity
def init_params(N, device, scale=0.08):
    mu3 = ((torch.rand(N, 3, device=device) * 2 - 1) * 1.5).requires_grad_()
    log_s = torch.log(scale * torch.ones(N, 3, device=device)).requires_grad_()
    quat = torch.zeros(N, 4, device=device); quat[:, 0] = 1.0; quat.requires_grad_()
    color = torch.zeros(N, 3, device=device, requires_grad=True)
    op_raw = torch.full((N,), -2.0, device=device, requires_grad=True)
    return [mu3, log_s, quat, color, op_raw]

def make_opt(P, lr=1e-2):
    return torch.optim.Adam([{"params": [p], "lr": lr} for p in P])   # 每个参数一组，P8 要用

@torch.no_grad()
def densify_3d(P, opt, g, budget, size_threshold=0.06, split_scale=1.6,
               prune_opacity=0.005, top_frac=0.25):
    mu3, log_s, quat, color, op_raw = P
    scale = log_s.exp()
    prune = op_raw.sigmoid() < prune_opacity
    dense = (g > torch.quantile(g, 1 - top_frac)) & ~prune
    room = budget - int((~prune).sum())
    if int(dense.sum()) > room:
        idx = torch.where(dense)[0]
        idx = idx[g[idx].argsort(descending=True)[:max(room, 0)]]
        dense = torch.zeros_like(dense); dense[idx] = True
    big = scale.max(dim=1).values > size_threshold
    split, clone = dense & big, dense & ~big
    keep = ~prune & ~split

    lg = math.log(split_scale)
    R = quaternion_to_rotation(quat[split])                  # (n, 3, 3)
    def child_mu():                                          # sample in parent gaussian's 3D sphere
        e = torch.randn_like(scale[split]) * scale[split]
        return mu3[split] + (R @ e[:, :, None]).squeeze(-1)
    extras = [
        torch.cat([mu3[clone],    child_mu(),         child_mu()]),
        torch.cat([log_s[clone],  log_s[split] - lg,  log_s[split] - lg]),
        torch.cat([quat[clone],   quat[split],        quat[split]]),
        torch.cat([color[clone],  color[split],       color[split]]),
        torch.cat([op_raw[clone], op_raw[split],      op_raw[split]]),
    ]
    return [swap_param(opt, grp, keep, ex) for grp, ex in zip(opt.param_groups, extras)]

def train_3d(cams, K, H, W, device, N0, steps, budget=None, densify_every=100, init_scale=0.08):
    torch.manual_seed(0)
    P = init_params(N0, device, init_scale)
    opt = make_opt(P)
    g_acc = torch.zeros(N0, device=device); n_acc = 0
    for step in range(steps):
        cam = cams[torch.randint(len(cams), (1,)).item()]
        img = render_view(P, cam, K, H, W)
        loss = ((img - cam["image"]) ** 2).mean()
        opt.zero_grad(); loss.backward()
        g_acc += P[0].grad.norm(dim=-1); n_acc += 1          # 3D positional gradient accumulated over random cameras
        opt.step()
        if budget and (step + 1) % densify_every == 0 and step < 0.7 * steps:
            P = densify_3d(P, opt, g_acc / n_acc, budget)
            g_acc = torch.zeros(P[0].shape[0], device=device); n_acc = 0
        if (step + 1) % 200 == 0:
            print(f"step {step + 1}  N {P[0].shape[0]}  loss {loss.item():.5f}", flush=True)
    return P

def rot_about(axis, ang):
    x, y, z = (axis / axis.norm()).tolist()
    Kx = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], device=axis.device)
    return torch.eye(3, device=axis.device) + math.sin(ang) * Kx + (1 - math.cos(ang)) * (Kx @ Kx)


if __name__ == "__main__":
    # device = get_device()
    # os.makedirs("results", exist_ok=True)
    # K, H, W, train, val = load_scene("data/spheres", device)
    # steps, budget = 3000, 2000

    # runs = {"p7": dict(N0=budget, budget=None),                      # fixed budget of 2000
    #         "p8": dict(N0=200, budget=budget, init_scale=0.15)}      # densify with starting point of 200
    # for tag, kw in runs.items():
    #     print("==", tag)
    #     P = train_3d(train, K, H, W, device, steps=steps, **kw)
    #     tr = evaluate(P, train, K, H, W, prefix=f"results/{tag}_train")
    #     va = evaluate(P, val,   K, H, W, prefix=f"results/{tag}_val")
    #     print(f"{tag}: N = {P[0].shape[0]}  train PSNR {tr:.2f}  val PSNR {va:.2f}")
    #     torch.save([p.detach().cpu() for p in P], f"results/{tag}_params.pt")

    device = get_device()
    K, H, W, train, val = load_scene("data/spheres", device)
    P = [p.to(device) for p in torch.load("results/p8_params.pt")]
    up = torch.stack([-c["R"][1] for c in train]).mean(0)      # 由训练相机估计世界的“上”方向
    frames = []
    for i in range(12):
        cam = {"R": val[0]["R"] @ rot_about(up, 2 * math.pi * i / 12), "t": val[0]["t"]}
        with torch.no_grad():
            img = render_view(P, cam, K, H, W).clamp(0, 1)
        save_img(img, f"results/p9_orbit_{i:02d}.png")
        frames.append(Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)))
    frames[0].save("results/p9_orbit.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)














