#!/usr/bin/env python3
"""
Render / preview SERL Franka sim image demos stored as a pickle file.

This repo's `franka_lift_cube_image_20_trajs.pkl` contains *transitions* (steps),
not a list of 20 trajectories. Episode boundaries are marked by `dones=True`.

What this script does
- Loads the pickle transitions
- Splits them into episodes by `dones`
- Renders a chosen episode from camera `front`, `wrist`, or both (side-by-side)
- Optionally exports to GIF/MP4 (if your environment has the needed deps)

Notes
- Observations are dicts containing image keys like: `front`, `wrist` (uint8, 1xHxWx3)
- This script only visualizes images; it doesn't need Mujoco / gym.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Episode:
    transitions: List[Dict[str, Any]]

    @property
    def length(self) -> int:
        return len(self.transitions)


def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert common image formats to uint8 RGB.
    Accepts shapes: (H,W,3), (1,H,W,3), (T,H,W,3) (uses first frame if needed).
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected numpy array image, got: {type(img)}")

    if img.ndim == 4:
        # (1,H,W,3) or (T,H,W,3) -> take first
        img = img[0]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected image shape (H,W,3) (or with leading batch), got: {img.shape}")

    if img.dtype == np.uint8:
        return img

    # float images
    img_f = img.astype(np.float32)
    # Heuristic: if in [0,1], scale; else clip to [0,255]
    if np.nanmax(img_f) <= 1.5:
        img_f = img_f * 255.0
    img_f = np.clip(img_f, 0.0, 255.0)
    return img_f.astype(np.uint8)


def _concat_horiz(images: Sequence[np.ndarray]) -> np.ndarray:
    if len(images) == 0:
        raise ValueError("No images to concatenate.")
    hws = [(im.shape[0], im.shape[1]) for im in images]
    if len(set(hws)) != 1:
        raise ValueError(f"All images must have same HxW to concat; got: {hws}")
    return np.concatenate(list(images), axis=1)


def load_transitions(pkl_path: Path) -> List[Dict[str, Any]]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or (len(data) > 0 and not isinstance(data[0], dict)):
        raise ValueError(
            "Unexpected pickle format. Expected a list[dict] of transitions "
            "with keys like observations/actions/dones."
        )
    return data


def split_episodes(transitions: Iterable[Dict[str, Any]]) -> List[Episode]:
    episodes: List[Episode] = []
    cur: List[Dict[str, Any]] = []

    for t in transitions:
        cur.append(t)
        if bool(t.get("dones", False)):
            episodes.append(Episode(transitions=cur))
            cur = []

    if cur:
        # If file ends without done=True, keep the tail as an episode.
        episodes.append(Episode(transitions=cur))
    return episodes


def infer_image_keys(transition: Dict[str, Any]) -> List[str]:
    obs = transition.get("observations", {})
    if not isinstance(obs, dict):
        return []

    keys: List[str] = []
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim in (3, 4) and v.shape[-1] == 3:
            keys.append(str(k))
    return sorted(set(keys))


def frame_from_transition(
    transition: Dict[str, Any],
    cameras: Sequence[str],
    use_next_obs: bool = False,
) -> np.ndarray:
    key = "next_observations" if use_next_obs else "observations"
    obs = transition.get(key)
    if not isinstance(obs, dict):
        raise ValueError(f"Transition missing dict `{key}`.")

    imgs: List[np.ndarray] = []
    for cam in cameras:
        if cam not in obs:
            raise KeyError(f"Camera key `{cam}` not found in {key}. Available: {sorted(obs.keys())}")
        imgs.append(_as_uint8_rgb(obs[cam]))

    if len(imgs) == 1:
        return imgs[0]
    return _concat_horiz(imgs)


def maybe_save_video(frames: Sequence[np.ndarray], out_path: Path, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()

    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Saving video requires `imageio`. Please install it in your environment."
        ) from e

    if suffix in (".gif", ".mp4"):
        if suffix == ".gif":
            imageio.mimsave(out_path.as_posix(), list(frames), duration=1.0 / max(fps, 1e-6))
        else:
            # MP4 typically needs `imageio-ffmpeg`.
            imageio.mimsave(out_path.as_posix(), list(frames), fps=fps)
        return

    raise ValueError(f"Unsupported output extension: {suffix}. Use .gif or .mp4")


def play_frames_matplotlib(frames: Sequence[np.ndarray], fps: float, title: str) -> None:
    import matplotlib.pyplot as plt

    interval = 1.0 / max(fps, 1e-6)
    fig = plt.figure(title)
    im = plt.imshow(frames[0])
    plt.axis("off")

    state = {"paused": False}

    def on_key(event):
        if event.key == " ":
            state["paused"] = not state["paused"]

    fig.canvas.mpl_connect("key_press_event", on_key)

    i = 0
    while i < len(frames):
        # If the window is closed, stop
        if not plt.fignum_exists(fig.number):
            break

        im.set_data(frames[i])
        
        # Check for pause state
        while True:
            if not plt.fignum_exists(fig.number):
                break
                
            status = " (PAUSED)" if state["paused"] else ""
            plt.title(f"{title}  |  step {i+1}/{len(frames)}{status}", fontsize=10)
            plt.pause(interval)
            
            if not state["paused"]:
                break
        
        if not plt.fignum_exists(fig.number):
            break
            
        i += 1

    # Keep window open unless running headless
    try:
        plt.show()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        type=str,
        default="franka_lift_cube_image_20_trajs.pkl",
        help="Path to the demo pickle file.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Which episode to render (0-based).",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="both",
        help="Camera to render: 'front', 'wrist', 'both', or a comma-separated list.",
    )
    parser.add_argument(
        "--use_next_obs",
        action="store_true",
        help="Render `next_observations` instead of `observations`.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Playback / export fps.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="If >0, only render the first N steps of the episode.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output file path (.gif or .mp4). If empty, no export.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open a matplotlib window (useful if you only export).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Demo pickle not found: {pkl_path.resolve()}")

    transitions = load_transitions(pkl_path)
    episodes = split_episodes(transitions)

    if len(episodes) == 0:
        raise RuntimeError("No episodes found (no transitions or missing `dones=True`).")

    ep_idx = int(args.episode)
    if ep_idx < 0 or ep_idx >= len(episodes):
        raise IndexError(f"--episode {ep_idx} out of range. Found {len(episodes)} episodes.")

    # Camera selection
    cam_arg = str(args.camera).strip().lower()
    available = infer_image_keys(episodes[ep_idx].transitions[0])
    if cam_arg == "both":
        cameras = [c for c in ("front", "wrist") if c in available] or available
    elif "," in cam_arg:
        cameras = [c.strip() for c in cam_arg.split(",") if c.strip()]
    else:
        cameras = [cam_arg]

    if not cameras:
        raise RuntimeError("No cameras selected.")

    # Build frames
    ep = episodes[ep_idx]
    max_steps = int(args.max_steps)
    n_steps = ep.length if max_steps <= 0 else min(ep.length, max_steps)
    frames = [
        frame_from_transition(ep.transitions[i], cameras=cameras, use_next_obs=bool(args.use_next_obs))
        for i in range(n_steps)
    ]

    title = f"{pkl_path.name} | ep {ep_idx}/{len(episodes)-1} | cams={','.join(cameras)} | len={ep.length}"
    print(title)
    print("episode lengths:", [e.length for e in episodes[: min(len(episodes), 30)]], "..." if len(episodes) > 30 else "")
    print("available image keys (first transition):", available)

    # Export
    if args.out:
        maybe_save_video(frames, Path(args.out), fps=float(args.fps))
        print(f"saved: {Path(args.out).resolve()}")

    # Show
    if not bool(args.no_show):
        play_frames_matplotlib(frames, fps=float(args.fps), title=title)


if __name__ == "__main__":
    # Avoid macOS QT backend quirks in some environments
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()


