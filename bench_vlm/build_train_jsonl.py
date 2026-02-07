# bench_vlm/build_train_jsonl.py

from __future__ import annotations

from config.config import CFG
from utils.io_utils import write_jsonl
from utils.data_builder import resolve_image_path, resolve_label_dirs, iter_label_files, load_label
from utils.prompt_builder_vlm import VlmPromptBuilder


def main():
    cfg = CFG
    cfg.ensure_dirs()

    pb = VlmPromptBuilder(task_scope=cfg.task_scope)
    prompt = pb.build_prompt()

    label_dirs = resolve_label_dirs(cfg.RUNS_DIR, cfg.dataset_name, cfg.label_run_tag)
    if not label_dirs:
        raise FileNotFoundError(
            f"label dirs not found: runs/{cfg.dataset_name}/{cfg.label_run_tag}/(train|valid|test)/outputs/labels"
        )

    train_rows = []
    valid_rows = []

    for info in label_dirs:
        files = iter_label_files(info.labels_dir)
        for fp in files:
            label = load_label(fp)
            row = {
                "image": resolve_image_path(cfg.ROOT, label["image"]),     # absolute or relative path 그대로
                "prompt": prompt,
                "target": pb.build_target(label),
            }
            if info.split == "train":
                train_rows.append(row)
            elif info.split == "valid":
                valid_rows.append(row)

    if cfg.max_train_samples is not None:
        train_rows = train_rows[:cfg.max_train_samples]
    if cfg.max_valid_samples is not None:
        valid_rows = valid_rows[:cfg.max_valid_samples]

    write_jsonl(cfg.train_jsonl, train_rows)
    write_jsonl(cfg.valid_jsonl, valid_rows)

    print(f"[ok] wrote train jsonl: {cfg.train_jsonl} ({len(train_rows)})")
    print(f"[ok] wrote valid jsonl: {cfg.valid_jsonl} ({len(valid_rows)})")


if __name__ == "__main__":
    main()
