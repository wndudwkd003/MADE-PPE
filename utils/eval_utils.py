# utils/eval_utils.py

from params.params import DatasetEnum, TestModeEnum, AgentEnum
from params.prompt_params import StageEnum
import json
import os
import re
from pathlib import Path

def get_test_targets(
    agent: AgentEnum,
    dataset: DatasetEnum,
    targets: list[int],
    run_dir: str,
):
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
        out[k] = total / count
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




def get_cached_image_path(
    original_path: str,
    cache_dir: Path
):
    filename = os.path.basename(original_path)
    base_name, _ = os.path.splitext(filename)

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
