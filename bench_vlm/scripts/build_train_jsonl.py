# benchmark_vlm/scripts/build_train_jsonl.py

from __future__ import annotations
import argparse
from bench_vlm.core.bench_config import BenchConfig
from bench_vlm.core.dataset import build_samples_from_runs
from bench_vlm.core.io_utils import write_jsonl

def dump_split(cfg: BenchConfig, split: str, out_path: str, max_n: int):
    samples = build_samples_from_runs(
        runs_dir=cfg.data.runs_dir,
        dataset=cfg.data.dataset,
        agent=cfg.data.agent,
        split=split,
        max_n=max_n,
    )
    rows = []
    for s in samples:
        rows.append({
            "image_path": s.image_path,
            "system": s.system,
            "user": s.user,
            "target": s.target_json,
        })
    write_jsonl(out_path, rows)
    print(f"[OK] {split}: {len(rows)} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = BenchConfig.load(args.config)

    dump_split(cfg, cfg.data.train_split, cfg.data.train_jsonl, cfg.data.max_train)
    dump_split(cfg, cfg.data.val_split, cfg.data.val_jsonl, cfg.data.max_val)
    dump_split(cfg, cfg.data.test_split, cfg.data.test_jsonl, cfg.data.max_test)


if __name__ == "__main__":
    main()