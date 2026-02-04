# worker/evaluation.py

from config.config import Config

from params.params import DatasetEnum, TestModeEnum, AgentEnum
from params.prompt_params import StageEnum
from pathlib import Path
import json
import os
from PIL import Image
from utils.clip_utils import clip_image_text_sims

import re
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import math


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


def run_evaluation(config: Config):
    test_dir = Path(config.eval_runs) / config.test_mode.value / config.dataset.value / config.agent.name
    test_dir.mkdir(parents=True, exist_ok=True)

    test_targets = config.test_targets
    print(f"Running evaluation on targets: {test_targets}")

    target_paths = get_test_targets(
        agent=config.agent,
        dataset=config.dataset,
        targets=test_targets,
        run_dir=config.runs,
    )

    print(f"[TEST] Number of target paths: {len(target_paths)}")

    if config.test_mode == TestModeEnum.MADE_PPE:
        test_made_ppe(config, target_paths, test_dir)

    elif config.test_mode == TestModeEnum.MADE_BENCH:
        test_made_bench(config, target_paths, test_dir)


    print("Evaluation completed.")


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





def test_made_ppe(config: Config, target_paths: list[Path], test_dir: Path):
    pass




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


def init_aggregate(test_keys: list[str]):
    return {k: {"total": 0.0, "count": 0} for k in test_keys}


def update_aggregate(aggregate: dict, key: str, score: float):
    aggregate[key]["total"] += float(score)
    aggregate[key]["count"] += 1


def finalize_aggregate(aggregate: dict):
    out = {}
    for k, v in aggregate.items():
        total = float(v["total"])
        count = int(v["count"])
        out[k] = (total / count) if count > 0 else 0.0
    return out


def dump_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def test_made_bench(config: Config, target_paths: list[Path], test_dir: Path):
    started_at = datetime.now().isoformat(timespec="seconds")

    # samples: {target_tag: {"train": [...], "valid": [...], "test": [...]} }
    samples = get_label_samples(target_paths)

    global_aggregate = init_aggregate(config.test_keys)
    per_target_scores = {}

    cache_dir = Path(config.cache_dir)

    for target_tag, target_samples in samples.items():
        print(f"\n[MADE-Bench] Evaluating target: {target_tag}")

        target_aggregate = init_aggregate(config.test_keys)

        for split, split_samples in target_samples.items():
            print(f"[MADE-Bench] Evaluating split: {split}, Number of samples: {len(split_samples)}")

            for sample in split_samples:
                image_path = sample["image"]
                image_path = get_cached_image_path(image_path, cache_dir)

                image = Image.open(image_path).convert("RGB")

                for test_key in config.test_keys:
                    texts = build_texts(sample, test_key)
                    if len(texts) == 0:
                        continue

                    scores = clip_image_text_sims(
                        image,
                        texts,
                        config.clip_vision,
                        config.clip_pretrained,
                        config.device,
                    )

                    avg_score = sum(scores) / len(scores)

                    update_aggregate(target_aggregate, test_key, avg_score)
                    update_aggregate(global_aggregate, test_key, avg_score)

        target_scores = finalize_aggregate(target_aggregate)
        per_target_scores[target_tag] = target_scores

    aggregate_scores = finalize_aggregate(global_aggregate)

    dump_json(
        test_dir / "summary.json",
        {
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "aggregate": aggregate_scores,
            "targets": per_target_scores,
            "meta": {
                "num_targets": len(samples),
                "dataset": config.dataset.name,
                "agent": config.agent.name,
                "test_mode": config.test_mode.value,
                "clip_vision": config.clip_vision,
                "clip_pretrained": config.clip_pretrained,
                "device": config.device,
            },
        },
    )

    # 전체 평균(논문용 대표값) bar 그래프 저장
    plot_bar(
        aggregate_scores,
        test_dir / "aggregate_bar.png",
        title="MADE-Bench scores (aggregate)",
    )

    # 타깃별 비교 grouped 그래프 저장
    plot_grouped(
        per_target_scores,
        keys=config.test_keys,
        out_path=test_dir / "grouped_by_target.png",
        title="MADE-Bench scores (grouped by target)",
    )

    print("\n[MADE-Bench] Final aggregated results:")
    for k in config.test_keys:
        v = aggregate_scores[k]
        print(f"  {k:20s}: {float(v):.4f}")

    print(f"\n[MADE-Bench] Saved summary:   {test_dir / 'summary.json'}")
    print(f"[MADE-Bench] Saved plot:      {test_dir / 'aggregate_bar.png'}")
    print(f"[MADE-Bench] Saved plot:      {test_dir / 'grouped_by_target.png'}")
