# utils/prompt_builder_single_oneshot.py

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Type, TypeVar

from params.prompt_params import PromptPack
from params.structure import WorkEnvironment, HazardFactor, PPEItem

E = TypeVar("E", bound=Enum)


def enum_keys(enum_cls: Type[E]):
    return [e.name for e in enum_cls]


class SingleOneShotPromptBuilder:
    """
    One-shot(1 call)로 5개 stage 결과를 모두 출력하게 만드는 prompt builder
    """

    def __init__(self):
        self.work_env_keys = enum_keys(WorkEnvironment)
        self.hazard_keys = enum_keys(HazardFactor)
        self.ppe_keys = enum_keys(PPEItem)

    # -----------------
    # public
    # -----------------
    def build(self, state: dict[str, Any] | None = None) -> PromptPack:
        state = state or {}
        return PromptPack(
            system=self.system_prompt(),
            user=self.user_prompt(state),
        )

    # -----------------
    # system
    # -----------------
    def system_prompt(self) -> str:
        return (
            "You are an industrial safety labeling assistant for Personal Protective Equipment (PPE) compliance. \n"
            "You will be given an image.\n"
            "Follow constraints and role instructions strictly.\n"
            "Keep reasons short and evidence-based.\n"
            "This is SINGLE-ONESHOT labeling: produce ALL stage outputs in ONE JSON.\n"
        )
    
    # -----------------
    # user
    # -----------------
    def user_prompt(self, state: dict[str, Any]) -> str:
        # 혹시 확장 가능하도록 state 입력은 열어둠
        state_json = json.dumps(state, ensure_ascii=False)

        return "\n".join([
            "[Constraint]",
            "1) Use ONLY the allowed enum keys listed below (exact match).",
            "2) You must internally follow the chain: WorkEnvironment -> Hazards -> Required PPE -> Wearing -> Improper Wearing.",
            "3) No over-recommendation: required_ppe must be justified by hazards and visible context.",
            "4) If uncertain, choose the most plausible option based on visible evidence and keep uncertainty implicit in reasons.",
            "5) Output must be a SINGLE JSON object matching OneShotOut schema exactly (no extra keys).",
            "",
            "Allowed labels:",
            f"- WorkEnvironment: {self.work_env_keys}",
            f"- HazardFactor: {self.hazard_keys}",
            f"- PPEItem: {self.ppe_keys}",
            "- Boolean: true/false (for worn)",
            "",
            "Proposal policy (for schema/mapping/label change):",
            "- Each proposal_* must follow ProposalOut exactly.",
            "- When no change needed: flag=false and set type/target_from/target_to/proposal to empty strings.",
            "",
            "[Internal procedure] (must do before final JSON)",
            "A) Decide exactly 1 work_environment key.",
            "B) Decide hazards list based on A and image evidence.",
            "C) Decide required_ppe list based on B and image evidence.",
            "D) For each PPE in required_ppe, set wearing[].worn true/false.",
            "E) improper_wearing: include ONLY PPE where wearing.worn=true.",
            "   - improper_wearing[].worn=true means worn but incorrectly; false means worn correctly.",
            "",
            "[Input]",
            f"State (optional): {state_json}",
            "",
            "[Task]",
            "Return OneShotOut JSON with keys:",
            "- work_environment, work_environment_reason, proposal_work_environment",
            "- hazards, hazards_reason, proposal_hazard",
            "- required_ppe, required_ppe_reason, proposal_compliance",
            "- wearing, wearing_reason",
            "- improper_wearing, improper_wearing_reason",
            "",
            "Only output JSON.",
        ])

    
