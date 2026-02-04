# utils/vis_utils.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_bar(scores: dict, out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(scores.keys())
    vals = [float(scores[k]) for k in keys]

    plt.figure(figsize=(10, 4))
    plt.bar(keys, vals)
    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_grouped(per_target_scores: dict, keys: list[str], out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    targets = list(per_target_scores.keys())
    if not targets:
        return

    x = np.arange(len(keys))
    width = 0.8 / max(len(targets), 1)

    plt.figure(figsize=(12, 5))

    for i, t in enumerate(targets):
        score_map = per_target_scores[t]
        vals = []
        for k in keys:
            v = score_map[k]
            vals.append(v)

        plt.bar(x + i * width - 0.4 + width / 2, vals, width, label=str(t))

    plt.ylim(0.0, 1.0)
    plt.title(title)
    plt.xticks(x, keys, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
