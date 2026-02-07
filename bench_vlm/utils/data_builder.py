# benchmark_vlm/utils/data_builder.py

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.io_utils import read_json


@dataclass
class LabelRunPaths:
    labels_dir: Path  # .../split/outputs/labels
    split: str


def resolve_image_path(root: Path, image_path: str | Path) -> str:
    p = Path(image_path)
    if not p.is_absolute():
        p = (root / p).resolve()
    return str(p)


def resolve_label_dirs(runs_dir: Path, dataset_name: str, run_tag: str) -> list[LabelRunPaths]:
    # runs/SH17/SINGLE_STEP1/train/outputs/labels
    base = runs_dir / dataset_name / run_tag
    out: list[LabelRunPaths] = []
    for split in ["train", "valid", "test"]:
        d = base / split / "outputs" / "labels"
        if d.exists():
            out.append(LabelRunPaths(labels_dir=d, split=split))
    return out


def iter_label_files(labels_dir: Path) -> list[Path]:
    return sorted(labels_dir.glob("*.json"))


def load_label(label_path: Path) -> dict[str, Any]:
    return read_json(label_path)
