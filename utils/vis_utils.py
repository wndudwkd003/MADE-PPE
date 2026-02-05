from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_bar(
    data: dict,
    out_path: Path,
    title: str,
    max_items: int | None = None,
    rotate_xticks: int = 60,
    x_label_fontsize: int = 9,
):
    items = list(data.items())
    items.sort(key=lambda x: float(x[1]), reverse=True)

    if max_items is not None:
        items = items[: int(max_items)]

    labels = [str(k) for k, _v in items]
    values = [float(v) for _k, v in items]

    plt.figure(figsize=(18, 5))
    bars = plt.bar(labels, values)
    plt.title(title)
    plt.xticks(rotation=rotate_xticks, ha="right", fontsize=x_label_fontsize)

    max_v = 0.0
    for v in values:
        if v > max_v:
            max_v = v

    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(v),
            f"{int(v)}" if float(v).is_integer() else f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylim(0.0, max_v * 1.15 + 1e-9)

    # 아래 여백을 늘려 x 라벨이 잘리지 않게 함
    plt.subplots_adjust(bottom=0.35)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close()

def plot_grouped(
    grouped: dict,
    keys: list[str],
    out_path: Path,
    title: str,
    normalize: bool = False,
    rotate_xticks: int = 45,
    x_label_fontsize: int = 9,
    show_total_on_top: bool = True,
):
    group_names = list(grouped.keys())
    group_names.sort()

    values_by_key = []
    totals = []

    for g in group_names:
        row = grouped[g]
        total = 0.0
        for k in keys:
            total += float(row[k])
        totals.append(float(total))

        per_key = []
        for k in keys:
            v = float(row[k])
            if normalize:
                if total > 0.0:
                    per_key.append(v / total)
                else:
                    per_key.append(0.0)
            else:
                per_key.append(v)
        values_by_key.append(per_key)

    xs = list(range(len(group_names)))

    plt.figure(figsize=(18, 5))
    bottom = [0.0 for _ in xs]

    for ki, k in enumerate(keys):
        heights = [values_by_key[i][ki] for i in range(len(group_names))]
        bars = plt.bar(xs, heights, bottom=bottom, label=str(k))

        if not normalize:
            for bar, h in zip(bars, heights):
                if h <= 0.0:
                    continue
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_y() + float(h) / 2.0,
                    f"{int(h)}" if float(h).is_integer() else f"{h:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        for i in range(len(bottom)):
            bottom[i] += float(heights[i])

    plt.title(title)
    plt.xticks(xs, group_names, rotation=rotate_xticks, ha="right", fontsize=x_label_fontsize)
    plt.legend()

    if normalize:
        plt.ylim(0.0, 1.0)
        if show_total_on_top:
            for i, total in enumerate(totals):
                plt.text(float(xs[i]), 1.01, f"n={int(total)}", ha="center", va="bottom", fontsize=9)
    else:
        max_total = 0.0
        for t in totals:
            if t > max_total:
                max_total = t
        plt.ylim(0.0, max_total * 1.15 + 1e-9)

        if show_total_on_top:
            for i, total in enumerate(totals):
                if total <= 0.0:
                    continue
                plt.text(float(xs[i]), float(total), f"{int(total)}", ha="center", va="bottom", fontsize=9)

    # 아래 여백을 크게 확보 (긴 라벨 대응)
    plt.subplots_adjust(bottom=0.40)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close()
