# benchmark_vlm/utils/train_utils.py

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
    out = []
    for r in rows:
        out.append(Sample(
            image=r["image"],
            prompt=r["prompt"],
            target=r["target"],
        ))
    return out


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
    Qwen2-VL (causal) SFT collator:
      - full_input = chat(user(image+prompt) + assistant(target))
      - labels = full_input_ids with prompt part masked to -100
    """
    def __init__(self, processor, max_prompt_tokens: int, max_target_tokens: int, max_image_size: int = 512):
        self.processor = processor
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens
        self.max_image_size = max_image_size

    def _resolve_image_path(self, p: str | Path) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return (CFG.ROOT / pp).resolve()

    def _load_image(self, p: str | Path) -> Image.Image:
        img = Image.open(self._resolve_image_path(p)).convert("RGB")
        img.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.BICUBIC)
        return img

    def _build_prompt_text(self, prompt: str) -> str:
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def _build_full_text(self, prompt: str, target: str) -> str:
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": target},
        ]
        return self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = [self._load_image(x["image"]) for x in batch]
        prompts = [x["prompt"] for x in batch]
        targets = [x["target"] for x in batch]

        prompt_texts = [self._build_prompt_text(p) for p in prompts]
        full_texts = [self._build_full_text(p, t) for p, t in zip(prompts, targets)]

        prompt_enc = self.processor(
            images=images,
            text=prompt_texts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_tokens,
            return_tensors="pt",
        )
        prompt_lens = (prompt_enc["attention_mask"].sum(dim=1)).tolist()

        full_max_len = self.max_prompt_tokens + self.max_target_tokens
        enc = self.processor(
            images=images,
            text=full_texts,
            padding=True,
            truncation=True,
            max_length=full_max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        labels = input_ids.clone()

        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        enc["labels"] = labels
        return enc


def build_model_and_processor(pretrained_id: str, bf16: bool = True, fp16: bool = False):
    """
    Load model + processor for VLM SFT.
    Default: AutoModelForVision2Seq (works for Qwen2-VL).
    """
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
