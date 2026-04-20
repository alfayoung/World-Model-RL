# 2DGS Panorama Documentation

```shell
conda activate xsim
```

Single source of truth for the panorama pipeline.

Owner workflow: `scripts/image_collect.py --mode panoramic`

## 1) What Step 1 Produces

Run:

```bash
PYTHONPATH=third_party/LIBERO python scripts/image_collect.py \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --mode panoramic \
  --camera_name frontview \
  --panoramic_yaw_angles 18 \
  --panoramic_pitch_angles [0] \
  --multiple_radii [0.8,1.0,1.2] \
  --height_variations [0.0] \
  --use_radius_variation True \
  --use_height_variation False \
  --look_at_offset_std 0.0 \
  --output_dir datasets/2dgs_live/<RUN_NAME>/input
```

Expected output in `input/`:
- `54` panorama images total in the standard setup (`18` yaw angles × `3` radii × `1` pitch)
- panorama images like `frontview_r0.80_h+0.000_az000.00_pitch+00.00.png`
- `cameras.txt`
- `images.txt`
- `camera_poses.json`

Important:
- the standard April 3 rerun uses `54` images, not `540`
- Step 1 does **not** create `points3D.txt`
- Step 1 does **not** create `points3D.ply`
- so Step 1 output is **not yet** a complete COLMAP sparse dataset
- Step 1 panorama PNGs must use the standard LIBERO vertical-only flip convention; do not apply an extra horizontal flip

## 2) What Step 2 Produces

After collection, build sparse initialization from simulator depth:

```bash
PYTHONPATH=third_party/LIBERO python scripts/build_panorama_sparse_from_sim_depth.py \
  --dataset_dir datasets/2dgs_live/libero_spatial_t0_panorama_init_20260413 \
  --task_suite_name libero_spatial \
  --task_id 0 \
  --camera_name frontview \
  # --disable_workspace_crop \
  # --max_points 1000000
```

This step creates:
- `sparse/0/cameras.txt`
- `sparse/0/images.txt`
- `sparse/0/points3D.txt`

This is the required output for training.

Important:
- Step 2 creates `sparse/0/points3D.txt`
- Step 2 now applies a default workspace crop around the tabletop scene to avoid reconstructing the full room background
- Step 2 does **not** need to create `sparse/0/points3D.ply`
- 2DGS will auto-convert `sparse/0/points3D.txt` to `sparse/0/points3D.ply` on first train/load

Required manual check:

```bash
bash scripts/build.sh 2
```

This helper is only for inspection. It runs COLMAP analysis on `sparse/0` and exports a preview PLY at dataset root as `points3D.ply`. It is **not** the required training step.

## 3) Train And Render

```bash
# 3) train
cd third_party/2d-gaussian-splatting
python train.py -s ../../datasets/2dgs_live/<RUN_NAME> -m output/<MODEL_NAME>

# 4) render
python render.py -m output/<MODEL_NAME> -s ../../datasets/2dgs_live/<RUN_NAME>
```

## 4) Required Checks Before Training

- `input/` contains only panorama images named `frontview_r...az...png`
- `input/cameras.txt` focal length matches simulator truth
- `input/images.txt` uses world-to-camera pose
- `sparse/0/points3D.txt` exists before training
- `54` images are expected in the standard setup unless you intentionally run a denser sweep

## 5) Do Not Mix This In

`collect_sim_demo.py` outputs `frame_0000.png`, `frame_0001.png`, ...

Those are replay rollout frames, not panorama capture. Do not mix them into the panorama dataset.

## 6) Debug Conclusion

Previous failures came from two issues:
- incorrect intrinsics / pose export in old `image_collect.py`
- inconsistent sparse initialization reused from an older coordinate convention

Current rule:
- use `image_collect.py --mode panoramic`
- then build a fresh sparse init with `build_panorama_sparse_from_sim_depth.py`
- if a dataset was generated with the old 180-degree image flip, rerun both Step 1 and Step 2 before training
