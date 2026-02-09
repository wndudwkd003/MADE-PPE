# utils/prompt_builder_single_oneshot.py

import json
from enum import Enum
from typing import Any, Type, TypeVar

from params.prompt_params import PromptPack
from params.structure import WorkEnvironment, HazardFactor, PPEItem
from pydantic import BaseModel
from params.common_prompt import CommonPrompt

E = TypeVar("E", bound=Enum)

def enum_keys(enum_cls: Type[E]):
    return [e.name for e in enum_cls]

class SingleOneShotPromptBuilder:
    STRICT_CONSTRAINTS = CommonPrompt.STRICT_CONSTRAINTS
    VISIBILITY_RULE = CommonPrompt.VISIBILITY_RULE
    SCHEMA_PROPOSAL_RULE = CommonPrompt.SCHEMA_PROPOSAL_RULE
    PROPOSAL_POLICY_TEXT = CommonPrompt.PROPOSAL_POLICY_TEXT

    def __init__(self):
        self.work_env_keys = enum_keys(WorkEnvironment)
        self.hazard_keys = enum_keys(HazardFactor)
        self.ppe_keys = enum_keys(PPEItem)

    def build(self, state: dict[str, Any] | None, text_format: BaseModel) -> PromptPack:
        state = state or {}
        return PromptPack(
            system=self.system_prompt(),
            user=self.user_prompt(state, text_format),
        )

    def system_prompt(self) -> str:
        return CommonPrompt.SYSTEM_PROMPT + "This is SINGLE-ONESHOT labeling: produce ALL stage outputs in ONE JSON.\n"


    def get_text_format_schema(self, text_format: BaseModel):
        sc = text_format.model_json_schema()
        # OneShotOut은 nested 구조이므로 $defs 등을 포함한 전체 스키마가 중요함.
        # title만 제거하고 전체 구조 유지
        if "title" in sc:
            del sc["title"]
        return json.dumps(sc, indent=2, ensure_ascii=False)

    def user_prompt(self, state: dict[str, Any], text_format: BaseModel):
        state_json = json.dumps(state or {}, ensure_ascii=False)
        text_format_info = self.get_text_format_schema(text_format)

        parts = [
            self.constraint_block(text_format_info),
            self.internal_procedure_block(),
            self.input_block(state_json),
            self.task_block_nested_oneshot(),
        ]
        return "\n".join([p for p in parts if p])

    def input_block(self, state_json: str):
        return "[Input]\n" f"State (previous outputs): {state_json}\n"

    # -----------------
    # constraints
    # -----------------
    def constraint_block(self, text_format_info: str):
        parts = [
            "[Constraint]",
            *self.STRICT_CONSTRAINTS,
            f"6) FORMATTING: Follow the JSON schema below exactily: \n{text_format_info}",
            "",
            "Allowed labels:",
            f"- WorkEnvironment: {self.work_env_keys}",
            f"- HazardFactor: {self.hazard_keys}",
            f"- PPEItem: {self.ppe_keys}",
            "- Boolean: true/false",
            "",
            self.PROPOSAL_POLICY_TEXT
        ]
        return "\n".join(parts) + "\n"

    # -----------------
    # self-critique block (Simulates Pipeline)
    # -----------------
    def internal_procedure_block(self):
        return (
            "[Internal procedure]\n"
            "You must internally perform these steps SEQUENTIALLY before finalizing your JSON:\n"
            "1. Identify Work Environment (Visual).\n"
            "2. Identify Hazards (Visual) based on the environment.\n"
            "3. Determine Required PPE (Regulation) based on identified hazards. (Constraint #2)\n"
            "4. Check Wearing Status (Visual) for each required PPE.\n"
            "5. Detect Improper Wearing (Visual) for worn items.\n"
            "6. Challenge your own draft (Self-Correction for Hallucinations).\n"
            "7. Finalize the JSON output.\n"
        )

    # -----------------
    # task
    # -----------------
    def task_block_nested_oneshot(self):
        return (
            "[Task]\n"
            "- Output OneShotOut JSON with EXACTLY the structure defined in the schema above.\n"
            "\n"
            "- STEP 1: Decide 'work_environment'. Choose exactly 1 key.\n"
            f"  {self.SCHEMA_PROPOSAL_RULE}\n"
            "\n"
            "- STEP 2: Decide 'hazard'. Select all applicable HazardFactor keys.\n"
            "  * Note: Select ONLY hazards actually visible or strictly implied by the task.\n"
            "\n"
            "- STEP 3: Decide 'compliance' (Required PPE).\n"
            "  * Rule: Determine requirements based on the Hazards found in STEP 2.\n"
            "  * Note: If a unique PPE is required but missing in mapping, use 'proposals'.\n"
            "\n"
            "- STEP 4: Decide 'wearing'.\n"
            "  * Action: For each PPE in 'required_ppe', decide worn=true/false.\n"
            f"  {self.VISIBILITY_RULE}\n"
            "\n"
            "- STEP 5: Decide 'improper_wearing'.\n"
            "  * Constraint: Include ONLY items where worn=true in STEP 4.\n"
            "  * Action: Set worn=true if worn incorrectly, false if correct.\n"
            "\n"
            "- Provide a short, evidence-based reason for each sub-object.\n"
        )
