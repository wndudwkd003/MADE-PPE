# benchmark_vlm/utils/prompt_builder_vlm.py

from __future__ import annotations

import json
from typing import Any
from params.params import TaskType


class VlmPromptBuilder:
    def __init__(self, task_mode: TaskType = TaskType.ALL_5STAGE):
        self.task_mode = task_mode
        # Task별 전용 질문 정의
        self.prompts = {
            TaskType.SCENE: "Describe the specific industrial work environment in this image.",
            TaskType.HAZARD: "Identify potential safety hazard factors in this work scene.",
            TaskType.REQUIRED: "List the mandatory Personal Protective Equipment (PPE) required for this situation.",
            TaskType.WEARING: "Analyze whether the workers are wearing the required PPE.",
            TaskType.IMPROPER: "Detect any improperly worn PPE (e.g., chin strap unbuckled)."
        }

    def build_prompt(self) -> str:
        if self.task_mode == TaskType.ALL_5STAGE:
            return "You are an industrial safety assistant. Output a JSON with work_environment, hazards, required_ppe, wearing, and improper_wearing."
        
        # 개별 Task 프롬프트
        instruction = self.prompts.get(self.task_mode, "Analyze this safety image.")
        return (
            f"{instruction}\n"
            "Output the result as a valid JSON object only."
        )

    def build_target(self, label: dict) -> str:
        # Task별로 필요한 key만 추출하여 Target 생성
        mapping = {
            TaskType.SCENE: ["work_environment"],
            TaskType.HAZARD: ["hazards"],
            TaskType.REQUIRED: ["required_ppe"],
            TaskType.WEARING: ["wearing"],
            TaskType.IMPROPER: ["improper_wearing"]
        }
        
        if self.task_mode == TaskType.ALL_5STAGE:
            keys = ["work_environment", "hazards", "required_ppe", "wearing", "improper_wearing"]
        else:
            keys = mapping.get(self.task_mode, [])

        target_obj = {k: label.get(k) for k in keys}
        return json.dumps(target_obj, ensure_ascii=False)