# benchmark_vlm/params/params.py

from __future__ import annotations
from enum import Enum


class LabelSource(str, Enum):
    MADE = "MADE"
    SINGLE_STEP = "SINGLE_STEP"
    SINGLE_ONESHOT = "SINGLE_ONESHOT"


class ModelName(str, Enum):
    QWEN2_VL = "qwen2_vl"
    # LLAVA = "llava"
    # BLIP2 = "blip2"
