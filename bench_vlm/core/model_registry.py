# benchmark_vlm/core/model_registry.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any

import torch


@dataclass
class VLMComponents:
    model: Any
    processor: Any


def _load_qwen2vl(pretrained: str, attn_impl: str = "sdpa"):
    """
    Qwen2-VL 계열 (HF Transformers).
    - 버전에 따라 class명이 달라질 수 있어 Auto*를 우선 사용.
    """
    from transformers import AutoProcessor, AutoModelForVision2Seq

    processor = AutoProcessor.from_pretrained(pretrained, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        pretrained,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        device_map="auto",
    )
    return VLMComponents(model=model, processor=processor)


REGISTRY = {
    # name -> loader
    "qwen2vl": _load_qwen2vl,
    # 여기에 "internvl", "llava", ... 계속 추가 가능
}

def load_vlm(name: str, pretrained: str, attn_impl: str = "sdpa") -> VLMComponents:
    if name not in REGISTRY:
        raise ValueError(f"Unknown model registry name: {name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](pretrained=pretrained, attn_impl=attn_impl)


def maybe_apply_lora(model, r: int, alpha: int, dropout: float, target_modules: Optional[list[str]]):
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, cfg)