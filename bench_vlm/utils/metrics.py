# bench_vlm/utils/metrics.py

from __future__ import annotations

import json
from typing import Any

from sklearn.metrics import f1_score


def safe_json_loads(s: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        obj = json.loads(s)
        return obj, None
    except Exception as e:
        return None, str(e)


def _as_sorted_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return sorted(x)
    return [x]


def _binary_acc(pred: Any, gold: Any) -> float:
    return 1.0 if pred == gold else 0.0


def _multi_label_f1(pred: list[str], gold: list[str]) -> float:
    labels = sorted(set(pred) | set(gold))
    if not labels:
        return 1.0
    y_true = [1 if l in gold else 0 for l in labels]
    y_pred = [1 if l in pred else 0 for l in labels]
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def eval_one(pred_obj: dict[str, Any], gold_obj: dict[str, Any], task_mode: str) -> dict[str, float]:
    out: dict[str, float] = {}

    out["work_environment_acc"] = _binary_acc(pred_obj.get("work_environment"), gold_obj.get("work_environment"))
    if task_mode == "scene":
        return out

    pred_h = _as_sorted_list(pred_obj.get("hazards"))
    gold_h = _as_sorted_list(gold_obj.get("hazards"))
    out["hazards_f1"] = _multi_label_f1(pred_h, gold_h)
    if task_mode == "hazard":
        return out

    pred_p = _as_sorted_list(pred_obj.get("required_ppe"))
    gold_p = _as_sorted_list(gold_obj.get("required_ppe"))
    out["required_ppe_f1"] = _multi_label_f1(pred_p, gold_p)
    if task_mode == "required":
        return out

    pred_w = pred_obj.get("wearing") or {}
    gold_w = gold_obj.get("wearing") or {}
    keys = sorted(set(pred_w.keys()) | set(gold_w.keys()))
    out["wearing_acc"] = 1.0 if not keys else sum(
        1.0 if bool(pred_w.get(k)) == bool(gold_w.get(k)) else 0.0 for k in keys
    ) / len(keys)
    if task_mode == "wearing":
        return out

    pred_i = pred_obj.get("improper_wearing") or {}
    gold_i = gold_obj.get("improper_wearing") or {}
    keys = sorted(set(pred_i.keys()) | set(gold_i.keys()))
    out["improper_wearing_acc"] = 1.0 if not keys else sum(
        1.0 if bool(pred_i.get(k)) == bool(gold_i.get(k)) else 0.0 for k in keys
    ) / len(keys)

    return out


def macro_f1_from_metrics(m: dict[str, float]) -> float:
    keys = [k for k in [
        "work_environment_acc",
        "hazards_f1",
        "required_ppe_f1",
        "wearing_acc",
        "improper_wearing_acc",
    ] if k in m]
    return 0.0 if not keys else sum(float(m[k]) for k in keys) / len(keys)


def avg_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}

    sums: dict[str, float] = {}
    for r in rows:
        for k, v in r.items():
            sums[k] = sums.get(k, 0.0) + float(v)

    avg = {k: v / len(rows) for k, v in sums.items()}
    avg["f1_macro"] = sum(macro_f1_from_metrics(r) for r in rows) / len(rows)
    return avg
