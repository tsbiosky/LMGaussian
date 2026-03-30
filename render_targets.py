"""
Render 2 target views from a trained LM-Gaussian model.

Uses Sim(3) alignment between original input poses (metadata.json) and
DUSt3R-estimated poses to transform the target camera poses into the
LMGaussian coordinate frame. DUSt3R applies a 100x scale internally.

Usage:
    python render_targets.py \
        -s data/custom_29 \
        --save outputs/custom_29 \
        --start_checkpoint outputs/custom_29/chkpnt30000.pth \
        --cameras_json /workspace/custom_Dataset/outputs/cameras.json \
        --metadata_json /workspace/custom_Dataset/inputs/metadata.json
"""
import os
import json
import math
import copy
import argparse

import torch
import torchvision
import numpy as np
from pathlib import Path

from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from arguments import ModelParams, PipelineParams, OptimizationParams360


DUST3R_SCALE = 100  # LMGaussian scales DUSt3R coords by 100x


def compute_sim3_alignment(src_positions, dst_positions):
    """Sim(3): dst ~ s * R @ src + t"""
    src = np.array(src_positions, dtype=np.float64)
    dst = np.array(dst_positions, dtype=np.float64)

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    src_s = np.sqrt(np.sum(src_c ** 2) / len(src))
    dst_s = np.sqrt(np.sum(dst_c ** 2) / len(dst))
    s = dst_s / src_s

    H = (src_c / src_s).T @ (dst_c / dst_s)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T

    t = dst_mean - s * R @ src_mean
    return s, R, t


def main():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    op = OptimizationParams360(parser)
    parser.add_argument("--start_checkpoint", type=str, required=True)
    parser.add_argument("--cameras_json", type=str, required=True)
    parser.add_argument("--metadata_json", type=str, required=True)
    args = parser.parse_args()

    dataset = lp.extract(args)
    pipe = pp.extract(args)
    opt = op.extract(args)

    with open(args.cameras_json) as f:
        cam_data = json.load(f)
    with open(args.metadata_json) as f:
        meta_data = json.load(f)

    target_c2ws = np.array(cam_data["camera_to_world"], dtype=np.float64)
    target_K = np.array(cam_data["camera_to_pixel"], dtype=np.float64)
    target_sizes = np.array(cam_data["image_size_xy"], dtype=np.float64)

    input_c2ws = np.array(meta_data["camera"]["camera_to_world"], dtype=np.float64)
    n_inputs = len(input_c2ws)
    print(f"Loaded {n_inputs} input poses and {len(target_c2ws)} target poses")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        dataset.model_path = args.save if hasattr(args, 'save') and args.save else dataset.model_path
        scene = Scene(dataset, gaussians, opt_depth=True, opt_normal=True, shuffle=False)
        gaussians.training_setup(opt)

        checkpoint = torch.load(args.start_checkpoint)
        model_params, _ = checkpoint
        gaussians.restore(model_params, opt)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        # Extract DUSt3R camera positions from train cameras
        train_cams = scene.getTrainCameras()
        dust3r_positions = []
        for cam in train_cams:
            w2c = getWorld2View2(cam.R, cam.T)
            c2w = np.linalg.inv(w2c)
            dust3r_positions.append(c2w[:3, 3])
        dust3r_positions = np.array(dust3r_positions)

        # Original input camera positions
        orig_positions = np.array([input_c2ws[i][:3, 3] for i in range(n_inputs)])

        print(f"Original positions range: {orig_positions.min(0)} to {orig_positions.max(0)}")
        print(f"DUSt3R positions range:   {dust3r_positions.min(0)} to {dust3r_positions.max(0)}")

        s, R_align, t_align = compute_sim3_alignment(orig_positions, dust3r_positions)
        print(f"Sim(3) alignment: scale={s:.4f}")

        aligned = s * (R_align @ orig_positions.T).T + t_align
        err = np.linalg.norm(aligned - dust3r_positions, axis=1)
        print(f"Alignment error: mean={err.mean():.4f}, max={err.max():.4f}")

        ref_cam = train_cams[0]
        save_dir = os.path.dirname(args.start_checkpoint)
        out_dir = os.path.join(save_dir, "target_renders")
        os.makedirs(out_dir, exist_ok=True)

        for i in range(len(target_c2ws)):
            c2w_orig = target_c2ws[i]
            # Transform target c2w into DUSt3R frame
            c2w_new = np.eye(4)
            c2w_new[:3, :3] = R_align @ c2w_orig[:3, :3]
            c2w_new[:3, 3] = s * R_align @ c2w_orig[:3, 3] + t_align
            w2c_new = np.linalg.inv(c2w_new)

            W = int(target_sizes[i][0])
            H = int(target_sizes[i][1])
            fx = target_K[i][0][0]
            fy = target_K[i][1][1]
            fov_x = 2.0 * math.atan(W / (2.0 * fx))
            fov_y = 2.0 * math.atan(H / (2.0 * fy))

            R_stored = w2c_new[:3, :3].T.astype(np.float32)
            t_w2c = w2c_new[:3, 3].astype(np.float32)

            dummy_img = torch.zeros((3, H, W), device="cuda")
            view = Camera(
                colmap_id=1000 + i,
                R=R_stored,
                T=t_w2c,
                FoVx=fov_x,
                FoVy=fov_y,
                image=dummy_img,
                gt_alpha_mask=None,
                gt_depth=None,
                gt_normal=None,
                image_name=f"output_{i}",
                uid=1000 + i,
            )

            rendering = render(view, gaussians, pipe, background, kernel_size=kernel_size)["render"]

            out_path = os.path.join(out_dir, f"output_{i}.png")
            torchvision.utils.save_image(rendering, out_path)
            print(f"Saved {out_path} ({W}x{H})")

    print(f"\nDone! Target renders saved to {out_dir}")


if __name__ == "__main__":
    main()
