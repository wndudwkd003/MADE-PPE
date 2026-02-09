# benchmark_vlm/params/params.py

from __future__ import annotations
from enum import Enum


class TaskType(str, Enum):
    """
    stage별 학습/평가 모드
    - SCENE      : 1) work_environment
    - HAZARD     : 1) + 2) hazards
    - REQUIRED   : 1) + 2) + 3) required_ppe
    - WEARING    : 1) + 2) + 3) + 4) wearing
    - IMPROPER   : 1) + 2) + 3) + 4) + 5) improper_wearing
    - ALL_5STAGE : 과거 one-shot 5단계 (= IMPROPER와 동일 키셋)
    """
    SCENE = "scene"
    HAZARD = "hazard"
    REQUIRED = "required"
    WEARING = "wearing"
    IMPROPER = "improper"
    ALL_5STAGE = "5stage"