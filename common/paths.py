"""Default assets and released models in the source-checkout layout."""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "models" / "g1"
CHECKPOINT_DIR = PROJECT_ROOT / "logs" / "rsl_rl"


def robot_asset_dir(robot: str = "g1", variant: str = "gripper") -> Path:
    if robot == "g1":
        return ASSET_DIR
    if robot == "oli" and variant in ("gripper", "hand"):
        return PROJECT_ROOT / "models" / "oli" / variant
    raise ValueError(f"Unsupported robot/variant: {robot}/{variant}")


def task_robot(task: str) -> str:
    return "oli" if "LimX-OLI" in task else "g1"


def mujoco_path(path: Path) -> str:
    """Prefer relative paths; MuJoCo's Windows file loader rejects some Unicode paths."""
    try:
        return os.path.relpath(path)
    except ValueError:  # Different Windows drives.
        return str(path)


def checkpoint_path(name: str, suffix: str = ".onnx") -> Path:
    """Resolve a released policy name to its local model path."""
    manifest = json.loads((CHECKPOINT_DIR / "manifest.json").read_text())
    for entry in manifest["models"]:
        if entry["name"] == name:
            return (CHECKPOINT_DIR / entry["path"]).with_suffix(suffix)
    raise ValueError(f"Unknown released policy: {name}")
