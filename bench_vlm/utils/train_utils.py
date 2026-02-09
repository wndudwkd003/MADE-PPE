# bench_vlm/utils/train_utils.py

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from utils.io_utils import read_jsonl
from config.config import CFG


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass
class Sample:
    image: str
    prompt: str
    target: str


def load_samples(jsonl_path: Path, max_samples: int | None = None) -> list[Sample]:
    rows = read_jsonl(jsonl_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    return [Sample(image=r["image"], prompt=r["prompt"], target=r["target"]) for r in rows]


class JsonlVlmDataset:
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        return {"image": s.image, "prompt": s.prompt, "target": s.target}


class VlmCollator:
    """
    Qwen2.5-VL SFT collator:
      user: [image] + prompt
      assistant: target(JSON)
    labels는 assistant 토큰만 학습하도록 prefix 부분 -100 처리
    """
    def __init__(self, processor, max_prompt_tokens: int, max_target_tokens: int, max_image_size: int | None):
        self.processor = processor
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens
        self.max_image_size = max_image_size

    def _resolve_image_path(self, p: str | Path) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (CFG.ROOT / pp).resolve()

    def _load_image(self, p: str) -> Image.Image:
        img = Image.open(self._resolve_image_path(p)).convert("RGB")
        if self.max_image_size is not None:
            img.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.BICUBIC)
        return img

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        images = [self._load_image(x["image"]) for x in batch]

        full_texts = []
        prefix_texts = []
        for x in batch:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": x["prompt"]},
                ],
            }
            assistant_msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": x["target"]}],
            }

            full_texts.append(
                self.processor.apply_chat_template([user_msg, assistant_msg], tokenize=False, add_generation_prompt=False)
            )
            prefix_texts.append(
                self.processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
            )

        prefix_inputs = self.processor(
            images=images,
            text=prefix_texts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens,
            return_tensors="pt",
        )
        full_inputs = self.processor(
            images=images,
            text=full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens + self.max_target_tokens,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs.get("attention_mask", torch.ones_like(input_ids))

        prefix_lens = prefix_inputs.get("attention_mask", torch.ones_like(prefix_inputs["input_ids"])).sum(dim=1)

        labels = input_ids.clone()
        for i, plen in enumerate(prefix_lens.tolist()):
            labels[i, :plen] = -100
        labels[attention_mask == 0] = -100

        full_inputs["labels"] = labels
        return full_inputs


def build_model_and_processor(pretrained_id: str, bf16: bool = True, fp16: bool = False):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq

    processor = AutoProcessor.from_pretrained(pretrained_id, trust_remote_code=True)

    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = "auto"

    model = AutoModelForVision2Seq.from_pretrained(
        pretrained_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    return model, processor


def maybe_apply_lora(model, use_lora: bool, r: int, alpha: int, dropout: float):
    if not use_lora:
        return model

    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except Exception:
        print("[warn] peft not installed; running full fine-tuning.")
        return model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, cfg)
