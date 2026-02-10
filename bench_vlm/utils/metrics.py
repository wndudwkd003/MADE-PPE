# bench_vlm/utils/metrics.py

from __future__ import annotations

import json
from typing import Any

from sklearn.metrics import f1_score


def safe_json_loads(s: str) -> tuple[dict[str, Any] | None, str | None]:
    """JSON 문자열을 안전하게 파싱합니다."""
    try:
        obj = json.loads(s)
        return obj, None
    except Exception as e:
        return None, str(e)


def _as_string_list(x: Any) -> list[str]:
    """
    입력값을 문자열 리스트로 변환합니다.
    None -> []
    List -> [str(item), str(item)...]
    Others -> [str(x)]
    """
    if x is None:
        return []
    if isinstance(x, list):
        # 딕셔너리나 숫자가 섞여있을 경우를 대비해 모두 문자열로 변환
        return [str(item) for item in x]
    return [str(x)]


def _parse_wearing_list(data: Any) -> dict[str, bool]:
    """
    wearing/improper_wearing 리스트를 딕셔너리로 변환합니다.
    입력 예시: [{"ppe": "A", "worn": true}, {"ppe": "B", "worn": false}]
    출력 예시: {"A": true, "B": false}
    """
    out = {}

    # 1. 데이터가 없으면 빈 딕셔너리 반환
    if not data:
        return out

    # 2. 리스트인 경우 (정상 케이스)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                # 키가 'ppe' 또는 'worn'이 있는지 확인
                ppe_code = item.get("ppe")
                is_worn = item.get("worn")

                # ppe 코드가 문자열이고 유효한 경우에만 추가
                if ppe_code and isinstance(ppe_code, str):
                    # is_worn이 bool이 아닐 경우(None, string 등) 처리
                    out[ppe_code] = bool(is_worn)

    # 3. 모델이 실수로 딕셔너리를 뱉은 경우 (Fallback)
    elif isinstance(data, dict):
        for k, v in data.items():
            out[str(k)] = bool(v)

    return out


def _binary_acc(pred: Any, gold: Any) -> float:
    return 1.0 if str(pred) == str(gold) else 0.0


def _multi_label_f1(pred: list[str], gold: list[str]) -> float:
    # 집합으로 변환하여 순서 무관하게 비교
    pred_set = set(pred)
    gold_set = set(gold)

    labels = sorted(pred_set | gold_set)
    if not labels:
        return 1.0

    y_true = [1 if l in gold_set else 0 for l in labels]
    y_pred = [1 if l in pred_set else 0 for l in labels]

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _wearing_acc(pred_dict: dict[str, bool], gold_dict: dict[str, bool]) -> float:
    """
    착용 상태 딕셔너리 두 개를 비교하여 정확도를 계산합니다.
    (PPE 종류와 착용 여부가 모두 일치해야 정답)
    """
    keys = sorted(set(pred_dict.keys()) | set(gold_dict.keys()))

    if not keys:
        return 1.0

    match_count = 0
    for k in keys:
        # 두 딕셔너리에 모두 키가 존재하고, 값(True/False)도 같아야 함
        val_pred = pred_dict.get(k)
        val_gold = gold_dict.get(k)

        # 키가 한쪽에만 있거나 값이 다르면 오답 (None과 False는 다르므로 주의)
        # 여기서는 "키가 없으면(None) False로 간주할지" 결정해야 하는데,
        # 데이터셋 정의상 명시되지 않은 PPE는 평가에서 제외하거나 False로 볼 수 있음.
        # 여기서는 엄격하게 '예측값과 정답값이 정확히 일치'하는지 봅니다.

        # 다만, 데이터셋에 없는 키를 모델이 예측했을 때 패널티를 주기 위해
        # get()의 기본값을 서로 다르게 주어 불일치하게 만듦
        if val_pred == val_gold:
            match_count += 1

    return float(match_count) / len(keys)


def eval_one(pred_obj: dict[str, Any], gold_obj: dict[str, Any], task_mode: str) -> dict[str, float]:
    out: dict[str, float] = {}

    # 1. Work Environment (String Exact Match)
    out["work_environment_acc"] = _binary_acc(
        pred_obj.get("work_environment"),
        gold_obj.get("work_environment")
    )
    if task_mode == "scene":
        return out

    # 2. Hazards (List of Strings F1)
    pred_h = _as_string_list(pred_obj.get("hazards"))
    gold_h = _as_string_list(gold_obj.get("hazards"))
    out["hazards_f1"] = _multi_label_f1(pred_h, gold_h)
    if task_mode == "hazard":
        return out

    # 3. Required PPE (List of Strings F1)
    pred_p = _as_string_list(pred_obj.get("required_ppe"))
    gold_p = _as_string_list(gold_obj.get("required_ppe"))
    out["required_ppe_f1"] = _multi_label_f1(pred_p, gold_p)
    if task_mode == "required":
        return out

    # 4. Wearing (List of Dicts -> Dict -> Accuracy)
    # [{"ppe": "A", "worn": true}] -> {"A": true} 변환
    pred_w = _parse_wearing_list(pred_obj.get("wearing"))
    gold_w = _parse_wearing_list(gold_obj.get("wearing"))
    out["wearing_acc"] = _wearing_acc(pred_w, gold_w)

    if task_mode == "wearing":
        return out

    # 5. Improper Wearing (List of Dicts -> Dict -> Accuracy)
    pred_i = _parse_wearing_list(pred_obj.get("improper_wearing"))
    gold_i = _parse_wearing_list(gold_obj.get("improper_wearing"))
    out["improper_wearing_acc"] = _wearing_acc(pred_i, gold_i)

    return out


def macro_f1_from_metrics(m: dict[str, float]) -> float:
    # 측정된 메트릭만 모아서 평균 계산
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
