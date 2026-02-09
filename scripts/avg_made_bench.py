import json
import os
from collections import defaultdict

# json 파일 경로 (여기에 경로 입력)
file_path = r"runs_eval/made_bench/SCP300/SINGLE_ONESHOT/summary.json"

if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    aggregate = data.get("aggregate", {})

    # 각 지표별로 값들을 모을 딕셔너리
    metric_values = defaultdict(list)
    all_values_flat = []

    # 데이터 순회하며 지표별로 값 수집
    for category_metrics in aggregate.values():
        for metric, value in category_metrics.items():
            val = float(value)
            metric_values[metric].append(val)
            all_values_flat.append(val)

    # 결과 계산
    result = {}

    # 1. 각 지표별(clip, blip 등) 평균 계산
    for metric, values in metric_values.items():
        result[metric] = sum(values) / len(values)

    # 2. 전체 데이터의 총 평균 (avg)
    if all_values_flat:
        result["avg"] = sum(all_values_flat) / len(all_values_flat)

    # 파일 저장 (_avg 추가)
    base, ext = os.path.splitext(file_path)
    output_path = f"{base}_avg{ext}"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
