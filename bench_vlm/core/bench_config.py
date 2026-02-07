# benchmark_vlm/core/bench_config.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
import yaml


@dataclass
class ModelCfg:
    name: str  # registry key
    pretrained: str  # HuggingFace model id
    local_path: Optional[str] = None

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None


@dataclass
class DataCfg:
    runs_dir: str = "runs"
    dataset: str = "SH17"         
    agent: str = "MADE"           
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"

    # optional caps
    max_train: int = -1
    max_val: int = -1
    max_test: int = -1

    # cache jsonl
    work_dir: str = "benchmark_vlm/workdir"
    train_jsonl: str = "benchmark_vlm/workdir/train.jsonl"
    val_jsonl: str = "benchmark_vlm/workdir/val.jsonl"
    test_jsonl: str = "benchmark_vlm/workdir/test.jsonl"


@dataclass
class TrainCfg:
    output_dir: str = "benchmark_vlm/ckpt"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    max_steps: int = -1

    # generation during eval
    gen_max_new_tokens: int = 256

    # precision
    bf16: bool = True
    fp16: bool = False

    # device / attention
    attn_implementation: str = "sdpa"  # "sdpa" or "flash_attention_2" (if installed)


@dataclass
class BenchConfig:
    seed: int
    model: ModelCfg
    data: DataCfg
    train: TrainCfg

    @staticmethod
    def load(path: str) -> "BenchConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def pick(dc_cls, d: dict[str, Any]):
            return dc_cls(**d)

        return BenchConfig(
            seed=int(raw.get("seed", 42)),
            model=pick(ModelCfg, raw["model"]),
            data=pick(DataCfg, raw["data"]),
            train=pick(TrainCfg, raw["train"]),
        )