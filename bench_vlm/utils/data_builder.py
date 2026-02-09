# bench_vlm/utils/data_builder.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.io_utils import read_json


@dataclass
class LabelRunPaths:
    labels_dir: Path  # .../split/outputs/labels
    split: str


def resolve_label_dirs(runs_dir: Path, dataset_name: str, run_tag: str) -> list[LabelRunPaths]:
    base = runs_dir / dataset_name / run_tag
    out: list[LabelRunPaths] = []
    for split in ["train", "valid", "test"]:
        d = base / split / "outputs" / "labels"
        if d.exists():
            out.append(LabelRunPaths(labels_dir=d, split=split))
    return out


def iter_label_files(labels_dir: Path) -> list[Path]:
    return sorted(labels_dir.glob("*.json"))


def resolve_image_path(image_path: str, root: Path) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str((root / p).resolve())


def _select_keys_for_mode(mode: str) -> list[str]:
    if mode == "scene":
        return ["image", "work_environment"]
    if mode == "hazard":
        return ["image", "work_environment", "hazards"]
    if mode == "required":
        return ["image", "work_environment", "hazards", "required_ppe"]
    if mode == "wearing":
        return ["image", "work_environment", "hazards", "required_ppe", "wearing"]
    if mode in ("improper", "5stage"):
        return ["image", "work_environment", "hazards", "required_ppe", "wearing", "improper_wearing"]
    raise ValueError(f"Unknown task_mode: {mode}")


def load_and_validate_label(label_path: Path, task_mode: str) -> dict[str, Any] | None:
    label = read_json(label_path)
    keep_keys = _select_keys_for_mode(task_mode)

    missing = [k for k in keep_keys if k not in label]
    if missing:
        print(f"[skip] missing keys in {label_path.name}: {missing}")
        return None

    trimmed = {k: label[k] for k in keep_keys}

    if task_mode in ("hazard", "required", "wearing", "improper", "5stage") and not isinstance(trimmed.get("hazards"), list):
        print(f"[skip] hazards is not list: {label_path.name}")
        return None
    if task_mode in ("required", "wearing", "improper", "5stage") and not isinstance(trimmed.get("required_ppe"), list):
        print(f"[skip] required_ppe is not list: {label_path.name}")
        return None
    if task_mode in ("wearing", "improper", "5stage") and not isinstance(trimmed.get("wearing"), dict):
        print(f"[skip] wearing is not dict: {label_path.name}")
        return None
    if task_mode in ("improper", "5stage") and not isinstance(trimmed.get("improper_wearing"), dict):
        print(f"[skip] improper_wearing is not dict: {label_path.name}")
        return None

    return trimmed
