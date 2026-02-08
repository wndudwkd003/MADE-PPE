# bench_vlm/config/config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from params.params import ModelName, LabelSource

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]  # .../MADE-PPE


@dataclass
class BenchmarkConfig:
    ROOT: Path = project_root()
    RUNS_DIR: Path = project_root() / "runs"
    DATASETS_DIR: Path = project_root() / "datasets"

    OUT_DIR: Path = project_root() / "bench_vlm" / "outputs"
    JSONL_BASE_DIR: Path = project_root() / "bench_vlm" / "outputs" / "jsonl"
    CKPT_DIR: Path = project_root() / "bench_vlm" / "outputs" / "checkpoints"
    EVAL_DIR: Path = project_root() / "bench_vlm" / "outputs" / "eval"

    dataset_name: str = "SH17"
    label_source: LabelSource = LabelSource.SINGLE_STEP
    label_run_tag: str = "SINGLE_STEP"

    task_scope: str = "5stage"

    model_name: ModelName = ModelName.QWEN2_VL
    pretrained_id: str = "Qwen/Qwen2-VL-2B-Instruct"

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

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    bf16: bool = True
    fp16: bool = False

    def jsonl_run_dir(self) -> Path:
        # outputs/jsonl/<label_run_tag>/
        return self.JSONL_BASE_DIR / self.label_run_tag

    @property
    def train_jsonl(self) -> Path:
        return self.jsonl_run_dir() / "train.jsonl"

    @property
    def valid_jsonl(self) -> Path:
        return self.jsonl_run_dir() / "valid.jsonl"

    def ensure_dirs(self):
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.JSONL_BASE_DIR.mkdir(parents=True, exist_ok=True)
        self.jsonl_run_dir().mkdir(parents=True, exist_ok=True)
        self.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        self.EVAL_DIR.mkdir(parents=True, exist_ok=True)


CFG = BenchmarkConfig()
