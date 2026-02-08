# benchmark_vlm/utils/metrics.py

from __future__ import annotations

import json
from typing import Any


def safe_json_loads(s: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return json.loads(s), None
    except Exception as e:
        return None, str(e)


def f1_set(pred: list[str], gold: list[str]) -> float:
    p = set(pred or [])
    g = set(gold or [])
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    inter = len(p & g)
    precision = inter / len(p)
    recall = inter / len(g)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def accuracy(a, b) -> float:
    return 1.0 if a == b else 0.0


def eval_one(pred_obj: dict[str, Any], gold_obj: dict[str, Any], task_scope: str) -> dict[str, float]:
    out = {}
    out["work_environment_acc"] = accuracy(pred_obj.get("work_environment"), gold_obj.get("work_environment"))
    out["hazards_f1"] = f1_set(pred_obj.get("hazards", []), gold_obj.get("hazards", []))
    out["required_ppe_f1"] = f1_set(pred_obj.get("required_ppe", []), gold_obj.get("required_ppe", []))

    if task_scope == "5stage":
        # wearing/improper: ppe별 bool 정확도 (union 기준)
        def to_map(xs):
            m = {}
            for it in xs or []:
                m[it.get("ppe")] = bool(it.get("worn"))
            return m

        pw = to_map(pred_obj.get("wearing", []))
        gw = to_map(gold_obj.get("wearing", []))
        keys = set(pw.keys()) | set(gw.keys())
        if not keys:
            out["wearing_acc"] = 1.0
        else:
            correct = 0
            for k in keys:
                correct += 1 if bool(pw.get(k, False)) == bool(gw.get(k, False)) else 0
            out["wearing_acc"] = correct / len(keys)

        pi = to_map(pred_obj.get("improper_wearing", []))
        gi = to_map(gold_obj.get("improper_wearing", []))
        keys2 = set(pi.keys()) | set(gi.keys())
        if not keys2:
            out["improper_wearing_acc"] = 1.0
        else:
            correct = 0
            for k in keys2:
                correct += 1 if bool(pi.get(k, False)) == bool(gi.get(k, False)) else 0
            out["improper_wearing_acc"] = correct / len(keys2)

    return out


def avg_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    out = {}
    for k in keys:
        out[k] = sum(r[k] for r in rows) / len(rows)
    return out