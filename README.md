# 15-674 A2: Gaussian Splatting

A small differentiable Gaussian splatting renderer written in PyTorch: 2D image fitting (P1–P5) and 3D scene reconstruction from posed views (P6–P9).

## Files

| File | Contents |
|---|---|
| `gaussian_2d.py` | P1 covariance and Gaussian weight, P2 rasterizer, P3 image fitting, P4 densification routine, P5 quality-vs-count experiment (`__main__`) |
| `p4.py` | P4 experiment: densified fit vs. plain fixed-count fit |
| `gaussian_3d.py` | P6 3D covariance and projection, P7/P8 training and 3D densification routines, P9 orbit render (`__main__`) |
| `p7_p8.py` | P7 and P8 experiments: trains and evaluates the plain and the densified 3D fits |

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install torch torchvision numpy pillow matplotlib
```

A CUDA GPU is used automatically when available (results were produced on an NVIDIA T4 / L4).

## Data

Place the provided data as follows:

```
data/
  2d/
    coffee.png
    astronaut.png
    cat.png
  spheres/
    cameras.json
    train/*.png
    val/*.png
```

The 2D images are used at their native 256×256 resolution, without resizing. All outputs are written to `results/`.

## 2D (P1–P5)

### P1–P3

The building blocks are in `gaussian_2d.py`:

- **P1**: `covariance_2d`, `gaussian_weight`
- **P2**: `render` (front-to-back alpha compositing, vectorized with a cumulative product)
- **P3**: `fit` (N Gaussians, spread-out random initialization, MSE loss, Adam)

### P4: densification

```bash
python p4.py            # uses data/2d/coffee.png; pass another image name as an argument
```

Runs two fits of the same image for 2000 steps each:

- `densified`: starts from 256 Gaussians and grows to a budget of 1024, with a densification pass every 200 steps (clone / split / prune), disabled for the last 30% of training
- `fixed`: plain fit with 1024 Gaussians

Outputs: `results/p4_<image>_densified.png`, `results/p4_<image>_fixed.png`, and the final Gaussian count and PSNR of each run printed to the terminal.

The densification itself is `densify` in `gaussian_2d.py`; `swap_param` rebuilds each parameter tensor while carrying the Adam state of the surviving Gaussians.

### P5: quality vs. number of Gaussians

```bash
python gaussian_2d.py
```

Fits each of the three images with N = 256, 1024 and 4096 Gaussians (fixed count, same initialization, 2000 Adam steps, seed 0).

Outputs:

- `results/<image>_<N>.png`: final render for each run
- `results/<image>_target.png`: the target image
- `results/p5_psnr.json`: final PSNR for all nine runs
- `results/p5_psnr_vs_n.png`: PSNR vs. number of Gaussians

The 2D renderer evaluates every Gaussian at every pixel, so the N = 4096 runs are slow and memory-hungry; the full set of nine runs takes several hours on a single GPU.

## 3D (P6–P9)

### P6

In `gaussian_3d.py`: `quaternion_to_rotation`, `covariance_3d`, `project_gaussian`.

### P7 and P8: reconstruction and 3D densification

```bash
python p7_p8.py
```

Trains two models for 3000 steps each on the 44 training views:

- `p7`: plain fit with a fixed random cloud of 2000 Gaussians
- `p8`: densified fit, starting from 200 Gaussians and growing to a budget of 2000 (a pass every 100 steps, disabled for the last 30% of training)

Outputs:

- `results/p7_train_*.png`, `results/p8_train_*.png`: render (left) next to ground truth (right) for three training views
- `results/p7_val_*.png`, `results/p8_val_*.png`: the same for three held-out views
- `results/p7_params.pt`, `results/p8_params.pt`: trained parameters
- Gaussian count and mean training / held-out PSNR for each model, printed to the terminal

Each run takes about three minutes on a T4. The relevant routines in `gaussian_3d.py` are `init_params`, `render_view`, `train_3d`, `densify_3d` and `evaluate`.

The 3D renderer (`render` in `gaussian_3d.py`) splits the image into 32×32 tiles and composites only the Gaussians whose 3σ bounding box touches each tile. `render_dense` is the unculled reference version.

### P9: held-out evaluation and novel views

The held-out PSNR is reported by `p7_p8.py` above. For the orbit render, with `results/p8_params.pt` present:

```bash
python gaussian_3d.py
```

Renders a 12-frame orbit (30° steps) around the vertical axis of the scene with the densified P8 model, starting from the pose of the first held-out camera.

Outputs: `results/p9_orbit_00.png` … `results/p9_orbit_11.png` and `results/p9_orbit.gif`.

## Results

2D, final PSNR in dB (256×256 images):

| Image | N = 256 | N = 1024 | N = 4096 |
|---|---|---|---|
| coffee | 25.62 | 29.84 | 35.05 |
| cat | 26.51 | 29.22 | 34.95 |
| astronaut | 21.07 | 25.39 | 32.52 |

3D, mean PSNR in dB:

| Model | Gaussians | Train | Held-out |
|---|---|---|---|
| P7, fixed count | 2000 | 24.29 | 17.17 |
| P8, densified (200 → 2000) | 2000 | 26.57 | 14.61 |
