# utils/eval_utils.py
from unittest import result
from gradio import get_image
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




def get_formatted_text(field_text: str):
    parts = field_text.split("_")

    if len(parts) > 0 and re.fullmatch(r"W\d+", parts[0]) is not None:
        parts = parts[1:]

    parts = [p.lower() for p in parts]
    return " ".join(parts)



def build_texts(
    sample: dict,
):
    results = {}

    for key, value in sample.items():
        if key == StageEnum.WORK_ENVIRONMENT.value:
            env_text = get_formatted_text(value)
            results[key] = [f"this work is {env_text}."]

        elif key == StageEnum.HAZARD.value:
            results[key] = [
                f"this image contains a {get_formatted_text(v)} hazard."
                for v in value
            ]

        elif key == StageEnum.COMPLIANCE.value:
            results[key] = [
                f"this work requires {get_formatted_text(v)}."
                for v in value
            ]

        elif key == StageEnum.WEARING.value:
            wearings = []
            for item in value:
                ppe = get_formatted_text(item["ppe"])
                worn = item["worn"]
                wearings.append(
                    f"the worker is {'wearing' if worn else 'not wearing'} {ppe}."
                )
            results[key] = wearings

        elif key == StageEnum.IMPROPER_WEARING.value:
            wearings = []
            for item in value:
                ppe = get_formatted_text(item["ppe"])
                worn = item["worn"]
                wearings.append(
                    f"the worker is wearing {ppe} {'improperly' if worn else 'properly'}."
                )
            results[key] = wearings

    return results



def group_samples_by_split_and_image(samples_by_target: dict):

    grouped = {"train": {}, "valid": {}, "test": {}}

    for target_tag, target_splits in samples_by_target.items():
        for split, split_samples in target_splits.items():
            if split not in grouped:
                continue

            for sample in split_samples:
                image_path = sample["image"]

                image_id = get_image_id(image_path)

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
    for item in value:
        ppe = item["ppe"]
        worn = item["worn"]
        out[ppe] = worn
    return out



def ppe_union(maps: list[dict]):
    u = set()
    for m in maps:
        for ppe in m.keys():
            u.add(ppe)
    return u


def consensus_bool_for_ppe(maps: list[dict], ppe: str):
    true_count = 0
    for m in maps:
        if m[ppe]:
            true_count += 1

    return true_count > (len(maps) // 2)


def score_ppe_bool_list(samples_by_target: dict, key: str):
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

    for m in maps:
        for ppe in ppes:
            if ppe not in m:
                m[ppe] = False

    consensus_by_ppe = {}
    for ppe in ppes:
        consensus_by_ppe[ppe] = consensus_bool_for_ppe(maps, ppe)

    num_ppes = len(ppes)

    if num_ppes == 0:
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
            if m[ppe] == c:
                correct += 1

        score = float(correct) / float(num_ppes)
        per_target[target_tag] = score
        total += score


    sample_score = total / float(n)
    return sample_score, per_target, used_targets


def get_avg(scores: list[float]):
    if len(scores) == 0:
        return 0.0
    return sum(scores) / float(len(scores))


def calc_avg(stats: dict, ndigits: int = 5, as_str: bool = False):
    out = {}
    for f, m_dict in stats.items():
        out[f] = {}
        for m, data in m_dict.items():
            v = data["total"] / float(data["count"]) if data["count"] > 0 else 0.0
            out[f][m] = f"{v:.{ndigits}f}" if as_str else round(v, ndigits)
    return out
