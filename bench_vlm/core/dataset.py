# benchmark_vlm/core/dataset.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import os

from .io_utils import read_json, read_jsonl
from .prompt import SYSTEM_PROMPT, make_user_prompt

def labels_dir(runs_dir: str, dataset: str, agent: str, split: str) -> Path:
    return Path(runs_dir) / dataset / agent / split / "outputs" / "labels"


def iter_label_files(runs_dir: str, dataset: str, agent: str, split: str) -> list[Path]:
    d = labels_dir(runs_dir, dataset, agent, split)
    if not d.exists():
        return []
    return sorted(d.glob("*.json"))


def label_to_target(label_obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "work_environment": label_obj.get("work_environment"),
        "hazards": label_obj.get("hazards", []),
        "required_ppe": label_obj.get("required_ppe", []),
        "wearing": label_obj.get("wearing", []),
        "improper_wearing": label_obj.get("improper_wearing", []),
    }


@dataclass
class BenchSample:
    image_path: str
    system: str
    user: str
    target_json: dict[str, Any]


def build_samples_from_runs(
    runs_dir: str,
    dataset: str,
    agent: str,
    split: str,
    max_n: int = -1,
) -> list[BenchSample]:
    files = iter_label_files(runs_dir, dataset, agent, split)
    if max_n != -1:
        files = files[:max_n]

    out: list[BenchSample] = []
    user_prompt = make_user_prompt()

    for fp in files:
        obj = read_json(fp)
        img = obj.get("image")
        if not img:
            continue
        if not os.path.exists(img):
            continue

        out.append(
            BenchSample(
                image_path=str(img),
                system=SYSTEM_PROMPT,
                user=user_prompt,
                target_json=label_to_target(obj),
            )
        )
    return out


# ---------- HF Dataset wrapper ----------
class JsonlBenchDataset:
    """
    jsonl format:
      {"image_path": "...", "system": "...", "user": "...", "target": {...}}
    """
    def __init__(self, jsonl_path: str):
        self.rows = read_jsonl(jsonl_path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self.rows[idx]
        return {
            "image_path": r["image_path"],
            "system": r["system"],
            "user": r["user"],
            "target": r["target"],
        }