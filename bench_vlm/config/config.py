# benchmark_vlm/config/config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bench_vlm.params.params import ModelName, LabelSource


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]  # .../MADE-PPE


@dataclass
class BenchmarkConfig:
    # --------- project paths ----------
    ROOT: Path = project_root()
    RUNS_DIR: Path = project_root() / "runs"
    DATASETS_DIR: Path = project_root() / "datasets"

    # benchmark outputs
    OUT_DIR: Path = project_root() / "bench_vlm" / "outputs"
    JSONL_DIR: Path = project_root() / "bench_vlm" / "outputs" / "jsonl"
    CKPT_DIR: Path = project_root() / "bench_vlm" / "outputs" / "checkpoints"
    EVAL_DIR: Path = project_root() / "bench_vlm" / "outputs" / "eval"

    # --------- what labels to train on ----------
    dataset_name: str = "SH17"
    label_source: LabelSource = LabelSource.SINGLE_STEP
    label_run_tag: str = "SINGLE_STEP1"


    # --------- task scope ----------
    # "3stage" = work/hazard/required_ppe만 학습/평가
    # "5stage" = work/hazard/required_ppe + wearing + improper_wearing까지
    task_scope: str = "5stage"  # "3stage" or "5stage"

    # --------- model selection ----------
    model_name: ModelName = ModelName.QWEN2_VL
    pretrained_id: str = "Qwen/Qwen2-VL-2B-Instruct"

    # --------- training hyperparams ----------
    train_jsonl: Path = JSONL_DIR / "train.jsonl"
    valid_jsonl: Path = JSONL_DIR / "valid.jsonl"

    max_train_samples: int | None = None
    max_valid_samples: int | None = None

    seed: int = 42
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

    max_prompt_tokens: int = 1024
    max_target_tokens: int = 256
    generation_max_new_tokens: int = 256

    # LoRA (optional)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # dtype / device
    bf16: bool = True
    fp16: bool = False

    def ensure_dirs(self):
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.JSONL_DIR.mkdir(parents=True, exist_ok=True)
        self.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        self.EVAL_DIR.mkdir(parents=True, exist_ok=True)


CFG = BenchmarkConfig()
