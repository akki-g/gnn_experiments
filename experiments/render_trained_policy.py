"""
Render a trained MAPPO policy on a VMAS environment and save a video.

Usage (CLI):
    python -m experiments.render_trained_policy \\
        --config configs/vmas_simple_spread_mappo.yaml \\
        --checkpoint runs/vmas_simple_spread/<timestamp>/ckpt/best.pt \\
        [--out /path/to/output/dir]
    python -m experiments.render_trained_policy \\
        --config configs/vmas_simple_spread_mappo.yaml \\
        --checkpoint runs/vmas_simple_spread/<timestamp>/ckpt/best.pt \\
        --no-video

Function API (called from train_vmas_mappo.py at end of training):
    from experiments.render_trained_policy import render_trained_policy
    render_trained_policy(ckpt_path=..., out_dir=..., cfg=...)

Video output:
    PRIMARY  : policy.mp4  via system ffmpeg (subprocess)
    FALLBACK : policy.gif  via imageio if ffmpeg fails
"""

import os
import argparse
import json
import subprocess
import tempfile

import numpy as np
import torch

# Heavy imports at function-call time, not at module load, so the module
# can be imported safely even if these are slow.


def _downscale(rgb: np.ndarray, size: int) -> np.ndarray:
    """
    Resize an [H, W, 3] uint8 numpy array to [size, size, 3].

    Uses PIL.Image.BILINEAR for smooth downscaling.

    Parameters
    ----------
    rgb  : [H, W, 3] uint8 numpy array (e.g. 1400×1400 from VMAS render).
    size : target square edge length in pixels.

    Returns
    -------
    resized : [size, size, 3] uint8 numpy array.
    """
    from PIL import Image

    img = Image.fromarray(rgb)
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _write_video(frames: list, out_dir: str, fps: int) -> None:
    """
    Write a list of [H, W, 3] uint8 numpy frames to disk as video.

    Tries system ffmpeg first (produces policy.mp4).
    Falls back to imageio.mimwrite if ffmpeg is unavailable or fails
    (produces policy.gif).

    Parameters
    ----------
    frames  : list of [H, W, 3] uint8 numpy arrays.
    out_dir : directory to write the video artifact into.
    fps     : frames per second.
    """
    if not frames:
        raise ValueError("Cannot write a video with zero frames.")

    from PIL import Image

    mp4_path = os.path.join(out_dir, "policy.mp4")
    gif_path = os.path.join(out_dir, "policy.gif")

    # Primary: ffmpeg via temp PNG sequence
    tmp_dir = tempfile.mkdtemp(prefix="render_frames_")
    try:
        # Write frames as numbered PNGs
        for i, frame in enumerate(frames):
            frame_path = os.path.join(tmp_dir, f"frame_{i:05d}.png")
            Image.fromarray(frame).save(frame_path)

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate", str(fps),
                "-i", os.path.join(tmp_dir, "frame_%05d.png"),
                "-pix_fmt", "yuv420p",
                mp4_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"[render] Video saved to {mp4_path}")
    except Exception as e:
        print(f"[render] ffmpeg failed ({e}), falling back to imageio GIF.")
        try:
            import imageio
            imageio.mimwrite(gif_path, frames, fps=fps)
            print(f"[render] GIF saved to {gif_path}")
        except Exception as e2:
            print(f"[render] imageio fallback also failed: {e2}. No video artifact written.")
    finally:
        # Clean up temp PNG files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def render_trained_policy(
    ckpt_path: str,
    out_dir: str,
    cfg: dict,
    *,
    write_video: bool = True,
) -> None:
    """
    Load a checkpoint, roll out a greedy policy, and write a video.

    Writes to out_dir:
        policy.mp4  (primary — system ffmpeg, when write_video=True)
        policy.gif  (fallback — imageio, when write_video=True)
        eval_metrics.json

    Parameters
    ----------
    ckpt_path : path to a .pt checkpoint written by MAPPO.save().
    out_dir   : directory to write the video and eval_metrics.json.
    cfg       : full training config dict (loaded from YAML).
    write_video: if False, skip VMAS rendering and write eval metrics only.
    """
    # Local imports to keep module load light
    from backbone import MAPPO
    from comm.identity import IdentityComm
    from backbone.utils import get_device
    from envs.adapters import VMASAdapter

    device = get_device(prefer_cuda=(cfg["device"] != "cpu"))

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_ctor_cfg = dict(ckpt["config"])

    # Ensure device key matches the inference device (avoid double-passing)
    model_ctor_cfg["device"] = str(device)

    model = MAPPO(**model_ctor_cfg, comm_module=IdentityComm())
    model.load(ckpt_path, map_location=device)
    model.eval()

    eval_seed = int(cfg["seed"]) + int(cfg["render"]["eval_seed"])

    env = VMASAdapter(
        scenario=cfg["env"]["scenario"],
        num_envs=1,
        n_agents=cfg["env"]["num_agents"],
        max_steps=cfg["env"]["max_steps"],
        seed=eval_seed,
        device=str(device),
    )

    frames = []
    ep_returns = []

    num_episodes = int(cfg["render"]["episodes"])
    if num_episodes <= 0:
        raise ValueError("render.episodes must be >= 1")
    max_steps = cfg["env"]["max_steps"]
    frame_size = cfg["render"]["frame_size"]
    fps = cfg["render"]["fps"]

    for ep in range(num_episodes):
        obs, state = env.reset()
        ep_ret = 0.0
        for t in range(max_steps):
            with torch.no_grad():
                out = model.act(obs.to(device), state.to(device), deterministic=True)
            obs, state, reward, done, info = env.step(out["action"].cpu())
            # reward[0, 0]: first env, first agent (team reward is identical across agents)
            ep_ret += reward[0, 0].item()
            if write_video:
                rgb = env.render()
                frames.append(_downscale(rgb, frame_size))
            if bool(done[0].any()):
                break
        ep_returns.append(ep_ret)

    mean_return = float(np.mean(ep_returns))
    print(f"[render] Eval over {num_episodes} episodes: mean_return={mean_return:.3f}")

    os.makedirs(out_dir, exist_ok=True)

    # Write eval metrics
    eval_metrics = {
        "episodes": num_episodes,
        "mean_return": mean_return,
        "returns": ep_returns,
    }
    eval_metrics_path = os.path.join(out_dir, "eval_metrics.json")
    with open(eval_metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"[render] Eval metrics saved to {eval_metrics_path}")

    if write_video:
        _write_video(frames, out_dir, fps)
    else:
        print("[render] Skipped video writing (--no-video / write_video=False).")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Render a trained MAPPO policy on a VMAS environment."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file used during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "Output directory for video and eval_metrics.json. "
            "Defaults to the parent of the checkpoint's parent directory "
            "(i.e., <run_dir>/ when checkpoint is at <run_dir>/ckpt/best.pt)."
        ),
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help=(
            "Run deterministic evaluation and write eval_metrics.json without "
            "calling VMAS render. Useful on headless HPC nodes."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    from backbone.utils import load_yaml_config
    cfg = load_yaml_config(args.config)

    if args.out is not None:
        out_dir = args.out
    else:
        # Default: parent of the checkpoint's parent dir
        # e.g. runs/vmas_simple_spread/<ts>/ckpt/best.pt → runs/vmas_simple_spread/<ts>/
        out_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))

    render_trained_policy(
        ckpt_path=args.checkpoint,
        out_dir=out_dir,
        cfg=cfg,
        write_video=not args.no_video,
    )
