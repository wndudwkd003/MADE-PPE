# utils/eval_utils.py
from params.params import DatasetEnum, TestModeEnum, AgentEnum
from params.prompt_params import StageEnum
import json
import os
import re
from pathlib import Path
from math import ceil


def get_test_targets(
    agent: AgentEnum,
    dataset: DatasetEnum,
    targets: list[int],
    run_dir: str,
) -> list[Path]:
    # runs/SH17/MADE1, MADE2, ... 예시

    base = Path(run_dir) / dataset.value

    paths = []

    for i in targets:
        agent_dir = base / f"{agent.name}{i}"
        paths.append(agent_dir)

    return paths


def init_aggregate(test_keys: list[str]):
    return {k: {"total": 0.0, "count": 0} for k in test_keys}


def update_aggregate(aggregate: dict, key: str, score: float):
    aggregate[key]["total"] += float(score)
    aggregate[key]["count"] += 1


def finalize_aggregate(aggregate: dict):
    out = {}
    for k, v in aggregate.items():
        total = v["total"]
        count = v["count"]
        out[k] = total / count # if count > 0 else 0.0
    return out


def get_label_samples(target_paths: list[Path]):
    samples_by_target = {}

    for path in target_paths:
        m = re.search(r"(\d+)$", path.name)
        if m:
            target_tag = f"MADE{m.group(1)}"
        else:
            target_tag = path.name

        if target_tag not in samples_by_target:
            samples_by_target[target_tag] = {"train": [], "valid": [], "test": []}

        for split in samples_by_target[target_tag].keys():
            if not (path / split).exists():
                continue

            sample_dir = path / split / "outputs" / "labels"
            if not sample_dir.exists():
                continue

            for sample_file in sample_dir.glob("*.json"):
                with open(sample_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    samples_by_target[target_tag][split].append(data)

    return samples_by_target

def get_all_samples(target_path: Path):
    out = {"train": [], "valid": [], "test": []}

    for split in out.keys():
        all_dir = target_path / split / "outputs" / "all"

        files = list(all_dir.glob("*.json"))

        for fp in files:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            out[split].append(data)

    return out

def get_image_id(image_path: str):
    filename = os.path.basename(image_path)
    base, _ = os.path.splitext(filename)
    return base


def get_cached_image_path(
    original_path: str,
    cache_dir: Path
):
    base_name = get_image_id(original_path)

    for f in cache_dir.iterdir():
        if f.is_file() and f.name.startswith(base_name):
            return str(f)

    return original_path



def build_texts(
    sample: dict,
    test_key: str
):
    texts = []

    if test_key == StageEnum.WORK_ENVIRONMENT.value:
        texts.append(f"this is a workplace environment of {sample[test_key]}")

    elif test_key == StageEnum.HAZARD.value:
        hazards = sample[test_key]

        for h in hazards:
            texts.append(f"this image contains the hazard of {h}")

    elif test_key == StageEnum.COMPLIANCE.value:
        ppe_list = sample[test_key]

        for ppe in ppe_list:
            texts.append(f"this image requires {ppe} for safety compliance")

    elif test_key == StageEnum.WEARING.value:
        wearing_list = sample[test_key]

        for wearing in wearing_list:
            ppe = wearing["ppe"]
            worn = wearing["worn"]

            if worn:
                texts.append(f"a person is wearing {ppe}")
            else:
                texts.append(f"a person is not wearing {ppe}")

    elif test_key == StageEnum.IMPROPER_WEARING.value:
        improper_wearing_list = sample[test_key]

        for improper in improper_wearing_list:
            ppe = improper["ppe"]
            worn = improper["worn"]

            if worn:
                texts.append(f"a person is improperly wearing {ppe}")
            else:
                texts.append(f"a person is wearing {ppe} properly")


    return texts



def group_samples_by_split_and_image(samples_by_target: dict):
    """
    입력:
      samples_by_target[target_tag][split] = [sample, ...]
    출력:
      grouped[split][image_id][target_tag] = sample
    """
    grouped = {"train": {}, "valid": {}, "test": {}}

    for target_tag, target_splits in samples_by_target.items():
        for split, split_samples in target_splits.items():
            if split not in grouped:
                continue

            for sample in split_samples:
                if "image" not in sample:
                    continue

                image_path = sample["image"]
                filename = os.path.basename(image_path)
                image_id, _ = os.path.splitext(filename)

                if image_id not in grouped[split]:
                    grouped[split][image_id] = {}

                grouped[split][image_id][target_tag] = sample

    return grouped


def majority_label_and_count(values: list[str]):
    counts = {}
    for v in values:
        if v in counts:
            counts[v] += 1
        else:
            counts[v] = 1

    best_label = ""
    best_count = 0
    for label, cnt in counts.items():
        if cnt > best_count:
            best_label = label
            best_count = cnt

    return best_label, best_count


def score_work_environment(samples_by_target: dict, key: str):
    """
    samples_by_target[target_tag] = sample
    반환:
      sample_score: float (0~1)
      per_target: dict[target_tag] = float (0~1)
      used_targets: list[str]
    """
    values = []
    used_targets = []

    for target_tag, sample in samples_by_target.items():
        if key in sample:
            values.append(sample[key])
            used_targets.append(target_tag)

    n = len(values)
    if n < 2:
        return 0.0, {}, []

    mode_label, mode_count = majority_label_and_count(values)

    per_target = {}
    for target_tag in used_targets:
        v = samples_by_target[target_tag][key]
        if v == mode_label:
            per_target[target_tag] = 1.0
        else:
            per_target[target_tag] = 0.0

    sample_score = float(mode_count) / float(n)
    return sample_score, per_target, used_targets


def list_to_set(value):
    out = set()
    if isinstance(value, list):
        for x in value:
            out.add(x)
    return out


def build_consensus_set(sets: list[set], threshold: int):
    freq = {}
    for s in sets:
        for item in s:
            if item in freq:
                freq[item] += 1
            else:
                freq[item] = 1

    consensus = set()
    for item, cnt in freq.items():
        if cnt >= threshold:
            consensus.add(item)

    return consensus


def f1_against_consensus(pred_set: set, consensus_set: set):
    if len(pred_set) == 0 and len(consensus_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(consensus_set) == 0:
        return 0.0

    inter = 0
    for x in pred_set:
        if x in consensus_set:
            inter += 1

    precision = float(inter) / float(len(pred_set))
    recall = float(inter) / float(len(consensus_set))
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


def score_set_list(samples_by_target: dict, key: str):
    """
    hazards, required_ppe 같은 list[str]에 대해 합의셋(consensus) 기반 F1 평균.
    반환:
      sample_score, per_target, used_targets
    """
    used_targets = []
    sets = []
    target_sets = {}

    for target_tag, sample in samples_by_target.items():
        if key in sample:
            s = list_to_set(sample[key])
            used_targets.append(target_tag)
            sets.append(s)
            target_sets[target_tag] = s

    n = len(sets)
    if n < 2:
        return 0.0, {}, []

    threshold = ceil(float(n) / 2.0)
    consensus = build_consensus_set(sets, threshold)

    per_target = {}
    total = 0.0
    for target_tag in used_targets:
        score = f1_against_consensus(target_sets[target_tag], consensus)
        per_target[target_tag] = score
        total += score

    sample_score = total / float(n)
    return sample_score, per_target, used_targets


def normalize_ppe_bool_list(value):
    """
    wearing / improper_wearing: [{"ppe": "...", "worn": bool}, ...] -> dict[ppe]=bool
    """
    out = {}
    if not isinstance(value, list):
        return out

    for item in value:
        if not isinstance(item, dict):
            continue
        if "ppe" not in item:
            continue
        if "worn" not in item:
            continue
        ppe = item["ppe"]
        worn = item["worn"]
        if isinstance(ppe, str) and isinstance(worn, bool):
            out[ppe] = worn

    return out


def ppe_union(maps: list[dict]):
    u = set()
    for m in maps:
        for ppe in m.keys():
            u.add(ppe)
    return u


def consensus_bool_for_ppe(maps: list[dict], ppe: str):
    t = 0
    f = 0
    for m in maps:
        if ppe in m:
            if m[ppe]:
                t += 1
            else:
                f += 1

    if t == 0 and f == 0:
        return None
    if t > f:
        return True
    if f > t:
        return False
    return None  # tie


def score_ppe_bool_list(samples_by_target: dict, key: str):
    """
    wearing / improper_wearing: PPE별 다수결 합의(consensus) 대비 타깃별 정확도 평균.
    tie는 그 PPE를 평가에서 제외.
    반환:
      sample_score, per_target, used_targets
    """
    used_targets = []
    maps = []
    target_maps = {}

    for target_tag, sample in samples_by_target.items():
        if key in sample:
            m = normalize_ppe_bool_list(sample[key])
            used_targets.append(target_tag)
            maps.append(m)
            target_maps[target_tag] = m

    n = len(maps)
    if n < 2:
        return 0.0, {}, []

    ppes = ppe_union(maps)

    consensus_by_ppe = {}
    num_considered = 0
    for ppe in ppes:
        c = consensus_bool_for_ppe(maps, ppe)
        if c is None:
            continue
        consensus_by_ppe[ppe] = c
        num_considered += 1

    if num_considered == 0:
        per_target = {}
        for target_tag in used_targets:
            per_target[target_tag] = 1.0
        return 1.0, per_target, used_targets

    per_target = {}
    total = 0.0

    for target_tag in used_targets:
        m = target_maps[target_tag]
        correct = 0
        for ppe, c in consensus_by_ppe.items():
            if ppe in m and m[ppe] == c:
                correct += 1
        score = float(correct) / float(num_considered)
        per_target[target_tag] = score
        total += score

    sample_score = total / float(n)
    return sample_score, per_target, used_targets
