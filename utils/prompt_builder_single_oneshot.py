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
    - params/output_schema.py의 OneShotOut (nested) 구조에 정확히 맞춘다.
    """

    def __init__(self):
        self.work_env_keys = enum_keys(WorkEnvironment)
        self.hazard_keys = enum_keys(HazardFactor)
        self.ppe_keys = enum_keys(PPEItem)

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
            "You are an industrial safety labeling assistant for Personal Protective Equipment (PPE) compliance.\n"
            "You will be given an image.\n"
            "Follow constraints and role instructions strictly.\n"
            "Keep reasons short and evidence-based.\n"
            "This is SINGLE-ONESHOT labeling: produce ALL stage outputs in ONE JSON.\n"
        )

    # -----------------
    # user
    # -----------------
    def user_prompt(self, state: dict[str, Any]):
        state_json = json.dumps(state or {}, ensure_ascii=False)

        parts = [
            self.constraint_block(),
            self.internal_procedure_block(),
            self.input_block(state_json),
            self.task_block_nested_oneshot(),
        ]
        return "\n".join([p for p in parts if p]) + "\n"

    def input_block(self, state_json: str):
        return "[Input]\n" f"State (previous outputs): {state_json}\n"

    # -----------------
    # constraints
    # -----------------
    def constraint_block(self):
        return "\n".join(
            [
                "[Constraint]",
                "1) Use ONLY the allowed labels listed below.",
                "2) If uncertain, choose the most plausible option based on visible evidence.",
                "3) Keep reasons short and evidence-based.",
                "4) Output MUST be a single JSON object that EXACTLY matches OneShotOut (nested).",
                "   - No extra text, no markdown, no extra keys.",
                "",
                "Allowed labels:",
                f"- WorkEnvironment: {self.work_env_keys}",
                f"- HazardFactor: {self.hazard_keys}",
                f"- PPEItem: {self.ppe_keys}",
                "- Boolean: true/false",
                "",
                self.proposal_policy_block(),
            ]
        )

    def proposal_policy_block(self):
        return (
            "Proposal policy (JUDGE only, for schema/mapping/label change):\n"
            "- proposal.flag:\n"
            "  - true  => you are proposing a change.\n"
            "  - false => no change needed.\n"
            "\n"
            "- proposal.type: choose EXACTLY ONE of the following (string):\n"
            "  - add | remove | modify\n"
            "\n"
            "- proposal.target_from:\n"
            "  - If type is modify or remove: MUST be an existing item in the current allowed label list.\n"
            "  - If type is add: use empty string \"\".\n"
            "\n"
            "- proposal.target_to:\n"
            "  - If type is add or modify: the desired new item/label key.\n"
            "  - If type is remove: use empty string \"\".\n"
            "\n"
            "- proposal.proposal: short evidence-based reason.\n"
            "\n"
            "- IMPORTANT:\n"
            "  - When proposal.flag=false, set: type=\"\", target_from=\"\", target_to=\"\", proposal=\"\".\n"
        )

    # -----------------
    # self-critique block
    # -----------------
    def internal_procedure_block(self):
        return (
            "[Internal procedure]\n"
            "You must internally perform these steps for EACH label group before finalizing your JSON:\n"
            "A) Proposer: draft the best candidate(s) using visible evidence.\n"
            "B) Rebutter: challenge your own draft (missing evidence? alternative interpretation?).\n"
            "C) Judge: finalize the answer that best satisfies constraints.\n"
            "D) If mapping/labels seem insufficient, use proposal fields (or keep them empty if no change).\n"
            "Only output the final JSON.\n"
        )

    # -----------------
    # task
    # -----------------
    def task_block_nested_oneshot(self):
        return (
            "[Task]\n"
            "- Output OneShotOut JSON with EXACTLY this NESTED structure:\n"
            "  {\n"
            "    \"work_environment\": {\n"
            "      \"work_environment\": <WorkEnvironment>,\n"
            "      \"reason\": <string>,\n"
            "      \"proposal\": <ProposalOut>\n"
            "    },\n"
            "    \"hazard\": {\n"
            "      \"hazards\": [<HazardFactor>, ...],\n"
            "      \"reason\": <string>,\n"
            "      \"proposal\": <ProposalOut>\n"
            "    },\n"
            "    \"compliance\": {\n"
            "      \"required_ppe\": [<PPEItem>, ...],\n"
            "      \"reason\": <string>,\n"
            "      \"proposal\": <ProposalOut>\n"
            "    },\n"
            "    \"wearing\": {\n"
            "      \"wearing\": [{\"ppe\": <PPEItem>, \"worn\": <bool>}, ...],\n"
            "      \"reason\": <string>\n"
            "    },\n"
            "    \"improper_wearing\": {\n"
            "      \"improper_wearing\": [{\"ppe\": <PPEItem>, \"worn\": <bool>}, ...],\n"
            "      \"reason\": <string>\n"
            "    }\n"
            "  }\n"
            "\n"
            "- Decide labels in this order (internally):\n"
            "  1) work_environment.work_environment: choose exactly 1 WorkEnvironment.\n"
            "  2) hazard.hazards: select all applicable HazardFactor.\n"
            "  3) compliance.required_ppe: decide FINAL PPE list.\n"
            "  4) wearing.wearing: for each PPE in required_ppe, set worn=true/false.\n"
            "  5) improper_wearing.improper_wearing: include ONLY PPE items that are worn=true in wearing.\n"
            "     - improper_wearing.worn=true means 'worn but incorrectly'; false means 'worn correctly'.\n"
            "\n"
            "- Provide a short reason for each sub-object.\n"
            "- proposal MUST be filled for work_environment / hazard / compliance (even if no change).\n"
        )
