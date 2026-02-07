# benchmark_vlm/utils/model_registry.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from params.params import ModelName


@dataclass
class ModelSpec:
    model_name: ModelName
    pretrained_id: str


def get_model_spec(model_name: ModelName, pretrained_id: str) -> ModelSpec:
    return ModelSpec(model_name=model_name, pretrained_id=pretrained_id)
