# worker/evaluation.py
import test
from params.prompt_params import StageEnum
from tqdm.auto import tqdm
from config.config import Config
from params.params import TestModeEnum
from pathlib import Path
from PIL import Image
from utils.sim_score_utils import ImageTextScorer
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
    group_samples_by_split_and_image,
    score_work_environment,
    score_set_list,
    score_ppe_bool_list,
    get_image_id,
    get_avg,
    calc_avg
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
    started_at = datetime.now().isoformat(timespec="seconds")

    samples = get_label_samples(target_paths)
    grouped = group_samples_by_split_and_image(samples)

    global_aggregate = init_aggregate(config.test_keys)

    per_target_aggregates = {}
    for target_tag in samples.keys():
        per_target_aggregates[target_tag] = init_aggregate(config.test_keys)

    counts = {
        "train": {"num_images": 0, "num_compared": 0},
        "valid": {"num_images": 0, "num_compared": 0},
        "test": {"num_images": 0, "num_compared": 0},
    }

    for split, images in grouped.items():
        counts[split]["num_images"] = len(images)

        print(f"\n[MADE-PPE] Evaluating split: {split}, Number of images: {len(images)}")

        for image_id, samples_by_target in images.items():
            counts[split]["num_compared"] += 1

            for test_key in config.test_keys:
                # [수정 1] 반환값에 weight 추가 (4개 변수로 언패킹)
                if test_key == StageEnum.WORK_ENVIRONMENT.value:
                    sample_score, per_target, used_targets, weight = score_work_environment(samples_by_target, test_key)

                elif test_key == StageEnum.HAZARD.value:
                    sample_score, per_target, used_targets, weight = score_set_list(samples_by_target, test_key)

                elif test_key == StageEnum.COMPLIANCE.value:
                    sample_score, per_target, used_targets, weight = score_set_list(samples_by_target, test_key)

                elif test_key == StageEnum.WEARING.value:
                    sample_score, per_target, used_targets, weight = score_ppe_bool_list(samples_by_target, test_key)

                elif test_key == StageEnum.IMPROPER_WEARING.value:
                    sample_score, per_target, used_targets, weight = score_ppe_bool_list(samples_by_target, test_key)

                else:
                    continue

                # [수정 2] update_aggregate 호출 시 weight 전달
                update_aggregate(global_aggregate, test_key, sample_score, weight)

                for target_tag in used_targets:
                    if target_tag in per_target:
                        # [수정 2] update_aggregate 호출 시 weight 전달
                        update_aggregate(per_target_aggregates[target_tag], test_key, per_target[target_tag], weight)

    print("\n[DEBUG] global_aggregate raw totals/counts:")
    for k in config.test_keys:
        v = global_aggregate[k]
        # [수정 3] total -> weighted_sum, total_weight로 키 변경 확인
        # init_aggregate에서 정의한 키와 일치해야 함
        print(f"  {k:30s} w_sum={v['weighted_sum']:.4f} t_weight={v['total_weight']:.4f} count={v['count']}")

    print("\n[DEBUG] StageEnum expected values:")
    print([
        StageEnum.WORK_ENVIRONMENT.value,
        StageEnum.HAZARD.value,
        StageEnum.COMPLIANCE.value,
        StageEnum.WEARING.value,
        StageEnum.IMPROPER_WEARING.value,
    ])

    aggregate_scores = finalize_aggregate(global_aggregate)


    per_target_scores = {}
    for target_tag, agg in per_target_aggregates.items():
        per_target_scores[target_tag] = finalize_aggregate(agg)

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
                "counts": counts,
                "test_keys": config.test_keys,
            },
        },
    )

    plot_bar(
        aggregate_scores,
        test_dir / "aggregate_bar.png",
        title="MADE-PPE agreement (aggregate)",
    )

    plot_grouped(
        per_target_scores,
        keys=config.test_keys,
        out_path=test_dir / "grouped_by_target.png",
        title="MADE-PPE agreement (by target)",
    )

    print("\n[MADE-PPE] Final aggregated results:")
    for k in config.test_keys:
        v = aggregate_scores[k]
        print(f"  {k:20s}: {float(v):.4f}")

    print(f"\n[MADE-PPE] Saved summary:   {test_dir / 'summary.json'}")
    print(f"[MADE-PPE] Saved plot:      {test_dir / 'aggregate_bar.png'}")
    print(f"[MADE-PPE] Saved plot:      {test_dir / 'grouped_by_target.png'}")

def test_made_bench(
    config: Config,
    target_paths: list[Path],
    test_dir: Path
):
    started_at = datetime.now().isoformat(timespec="seconds")
    samples_out_dir = test_dir / "samples"
    samples_out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["clip", "blip_itm", "blip_itc", "gme", "gme_inst"]
    target_stats = {}
    global_stats = {k: {m: {"total": 0.0, "count": 0} for m in metrics} for k in config.test_keys}

    scorer = ImageTextScorer(
        clip_model_name=config.clip_model,
        clip_pretrained=config.clip_pretrained,
        blip_model_name=config.blip_model,
        gme_model_name=config.gme_model,
        device=config.device,
    )

    samples = get_label_samples(target_paths)

    # 1. Target 단위에 tqdm 적용
    target_pbar = tqdm(samples.items(), desc="Targets", leave=True)
    for target_key, splits in target_pbar:
        target_pbar.set_postfix(target=target_key)

        target_stats[target_key] = {k: {m: {"total": 0.0, "count": 0} for m in metrics} for k in config.test_keys}
        target_sample_out_dir = samples_out_dir / target_key
        target_sample_out_dir.mkdir(parents=True, exist_ok=True)

        for split, split_samples in splits.items():
            sample_pbar = tqdm(split_samples, desc=f"  - {split}", leave=False)
            for sample in sample_pbar:
                image_path = sample["image"]
                image_id = get_image_id(image_path)

                cached_image_path = get_cached_image_path(image_path, Path(config.cache_dir))
                image = Image.open(cached_image_path).convert("RGB")

                formatted_texts = build_texts(sample)
                sample_results = {"image_id": image_id, "target": target_key, "split": split, "scores": {}}

                for field, texts in formatted_texts.items():
                    if len(texts) == 0 or field not in config.test_keys:
                        continue

                    # 모델 추론 (Heavy Task)
                    clip_sims = scorer.get_clip_score(image, texts)
                    blip_itm_score, blip_itc_score = scorer.get_blip_score(image, texts)
                    gme_score, gme_inst_score = scorer.get_gme_score(image, texts)

                    raw_scores = {
                        "clip": get_avg(clip_sims),
                        "blip_itm": get_avg(blip_itm_score),
                        "blip_itc": get_avg(blip_itc_score),
                        "gme": get_avg(gme_score),
                        "gme_inst": get_avg(gme_inst_score),
                    }

                    field_scores = {k: f"{v:.5f}" for k, v in raw_scores.items()}

                    sample_results["scores"][field] = field_scores

                    for m in metrics:
                        val = raw_scores[m]
                        target_stats[target_key][field][m]["total"] += val
                        target_stats[target_key][field][m]["count"] += 1
                        global_stats[field][m]["total"] += val
                        global_stats[field][m]["count"] += 1

                dump_json(
                    target_sample_out_dir / f"{target_key}_{image_id}_{split}.json",
                    sample_results,
                )

    aggregate_scores = calc_avg(global_stats, ndigits=5, as_str=True)
    per_target_scores = {tag: calc_avg(stats, ndigits=5, as_str=True) for tag, stats in target_stats.items()}

    dump_json(
        test_dir / "summary.json",
        {
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "aggregate": aggregate_scores,
            "targets": per_target_scores,
            "meta": {
                "num_targets": len(samples),
                "test_mode": config.test_mode.value,
                "test_keys": config.test_keys,
                "metrics": metrics
            },
        },
    )

    print(f"[MADE-BENCH] Evaluation finished. Summary saved to {test_dir / 'summary.json'}")






