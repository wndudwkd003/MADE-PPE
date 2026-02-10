# bench_vlm/config/config.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from params.params import TaskType

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]  # .../MADE-PPE


@dataclass
class BenchmarkConfig:
    ROOT: Path = project_root()
    RUNS_DIR: Path = project_root() / "runs"
    DATASETS_DIR: Path = project_root() / "datasets"

    OUT_DIR: Path = project_root() / "bench_vlm" / "outputs"

    dataset_name: str = "SCP300"
    label_run_tag: str = "MADE1"

    pretrained_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    task_mode: TaskType = TaskType.ALL_5STAGE   # TaskType : SCENE, HAZARD, REQUIRED, WEARING, IMPROPER, ALL_5STAGE

    seed: int = 42
    num_train_epochs: int = 3
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
    max_target_tokens: int = 1024
    generation_max_new_tokens: int = 1024

    image_size: int = 512

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    bf16: bool = True
    fp16: bool = False

    max_train_samples: int | None = None
    max_valid_samples: int | None = None


    @property
    def JSONL_DIR(self) -> Path:
        return self.OUT_DIR / "jsonl" / self.dataset_name / self.label_run_tag

    @property
    def CKPT_DIR(self) -> Path:
        return self.OUT_DIR / "checkpoints" / self.dataset_name / self.label_run_tag

    @property
    def EVAL_DIR(self) -> Path:
        return self.OUT_DIR / "eval" / self.dataset_name / self.label_run_tag

    @property
    def train_jsonl(self) -> Path:
        return self.JSONL_DIR / f"train_{self.task_mode.value}.jsonl"

    @property
    def valid_jsonl(self) -> Path:
        return self.JSONL_DIR / f"valid_{self.task_mode.value}.jsonl"

    def ensure_dirs(self):
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.JSONL_DIR.mkdir(parents=True, exist_ok=True)
        self.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        self.EVAL_DIR.mkdir(parents=True, exist_ok=True)


CFG = BenchmarkConfig()
