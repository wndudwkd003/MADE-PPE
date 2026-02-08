# benchmark_vlm/utils/prompt_builder_vlm.py

from __future__ import annotations

import json
from typing import Any


class VlmPromptBuilder:
    """
    VLM 학습/평가용 단일 프롬프트.
    - 입력: 이미지 1장
    - 출력: labels/*.json 과 동일한 key를 갖는 JSON
    """

    def __init__(self, task_scope: str = "5stage"):
        assert task_scope in ("3stage", "5stage")
        self.task_scope = task_scope

    def build_prompt(self) -> str:
        # 멀티 에이전트의 “proposer/rebutter/judge”를 VLM 학습에서는
        # "internally think"로만 유지하고 결과 JSON만 출력하게 유도

        keys_3 = ["work_environment", "hazards", "required_ppe"]
        keys_5 = keys_3 + ["wearing", "improper_wearing"]

        out_keys = keys_5 if self.task_scope == "5stage" else keys_3
        out_schema = {k: "..." for k in out_keys}

        return (
            "You are an industrial safety labeling assistant.\n"
            "Given ONE image, output a SINGLE JSON object that matches the required label keys.\n"
            "Rules:\n"
            "1) Output must be ONLY valid JSON. No extra text.\n"
            "2) Keep label keys EXACTLY as specified.\n"
            "3) hazards/required_ppe are list[str].\n"
            "4) wearing/improper_wearing are list of {\"ppe\": str, \"worn\": bool}.\n"
            "\n"
            f"Required JSON keys: {out_keys}\n"
            f"Example schema (structure only): {json.dumps(out_schema)}\n"
        )

    def build_target(self, label: dict[str, Any]) -> str:
        base = {
            "work_environment": label.get("work_environment"),
            "hazards": label.get("hazards", []),
            "required_ppe": label.get("required_ppe", []),
        }
        if self.task_scope == "5stage":
            base["wearing"] = label.get("wearing", [])
            base["improper_wearing"] = label.get("improper_wearing", [])
        return json.dumps(base, ensure_ascii=False)
