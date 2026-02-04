# worker/evaluation.py

from config.config import Config
from params.params import TestModeEnum
from pathlib import Path
from PIL import Image
from utils.clip_utils import clip_image_text_sims
from datetime import datetime
from utils.vis_utils import plot_bar, plot_grouped
from utils.eval_utils import (
    get_test_targets,
    init_aggregate,
    update_aggregate,
    build_texts,
    finalize_aggregate,
    get_cached_image_path,
    get_label_samples,
)
from utils.json_utils import dump_json



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



def test_made_ppe(config: Config, target_paths: list[Path], test_dir: Path):
    pass


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
