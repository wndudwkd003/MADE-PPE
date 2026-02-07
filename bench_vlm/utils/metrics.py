# benchmark_vlm/core/metrics.py

from __future__ import annotations
from typing import Any
import json

def _safe_set(xs):
    if xs is None:
        return set()
    return set([x for x in xs if x is not None])

def _wearing_to_map(items):
    # [{"ppe": "...", "worn": bool}, ...]
    m = {}
    for it in items or []:
        p = it.get("ppe")
        if p is None:
            continue
        m[p] = bool(it.get("worn"))
    return m

def _f1_from_counts(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

def eval_sample(gt: dict[str, Any], pred: dict[str, Any]) -> dict[str, float]:
    out = {}

    # 1) work_environment (single-label acc)
    out["work_env_acc"] = float(gt.get("work_environment") == pred.get("work_environment"))

    # 2) hazards (multi-label micro f1)
    gt_h = _safe_set(gt.get("hazards", []))
    pr_h = _safe_set(pred.get("hazards", []))
    tp = len(gt_h & pr_h)
    fp = len(pr_h - gt_h)
    fn = len(gt_h - pr_h)
    _, _, f1 = _f1_from_counts(tp, fp, fn)
    out["hazards_micro_f1"] = f1

    # 3) required_ppe (multi-label micro f1)
    gt_p = _safe_set(gt.get("required_ppe", []))
    pr_p = _safe_set(pred.get("required_ppe", []))
    tp = len(gt_p & pr_p)
    fp = len(pr_p - gt_p)
    fn = len(gt_p - pr_p)
    _, _, f1 = _f1_from_counts(tp, fp, fn)
    out["ppe_micro_f1"] = f1

    # 4) wearing (per-ppe accuracy over union)
    gt_w = _wearing_to_map(gt.get("wearing", []))
    pr_w = _wearing_to_map(pred.get("wearing", []))
    keys = set(gt_w.keys()) | set(pr_w.keys())
    if keys:
        correct = sum(1 for k in keys if gt_w.get(k) == pr_w.get(k))
        out["wearing_acc"] = correct / len(keys)
    else:
        out["wearing_acc"] = 1.0  # 둘 다 empty면 맞춘 걸로

    # 5) improper_wearing (per-ppe accuracy over union)
    gt_i = _wearing_to_map(gt.get("improper_wearing", []))
    pr_i = _wearing_to_map(pred.get("improper_wearing", []))
    keys = set(gt_i.keys()) | set(pr_i.keys())
    if keys:
        correct = sum(1 for k in keys if gt_i.get(k) == pr_i.get(k))
        out["improper_acc"] = correct / len(keys)
    else:
        out["improper_acc"] = 1.0

    # 6) overall (간단 평균)
    out["overall"] = (
        out["work_env_acc"]
        + out["hazards_micro_f1"]
        + out["ppe_micro_f1"]
        + out["wearing_acc"]
        + out["improper_acc"]
    ) / 5.0

    return out


def parse_pred_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    cand = text[start : end + 1]
    try:
        obj = json.loads(cand)
        for k in ["work_environment", "hazards", "required_ppe", "wearing", "improper_wearing"]:
            if k not in obj:
                return None
        return obj
    except Exception:
        return None