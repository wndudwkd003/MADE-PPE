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
    - ProposalOut는 'proposals: list[ProposalOut]' 형태로 반영한다.
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
            "Proposal policy (for label-set / mapping-table changes):\n"
            "\n"
            "- You must output proposals as a list field named: proposals\n"
            "- If no changes are needed, set proposals to an empty list: []\n"
            "\n"
            "Each item in proposals must match ProposalOut:\n"
            "- flag: must be true for every proposal item in the list\n"
            "\n"
            "- kind: choose EXACTLY ONE of the following (string):\n"
            "  - label | mapping\n"
            "\n"
            "- subject depends on kind:\n"
            "  - kind=label: work_environment | hazard | ppe\n"
            "  - kind=mapping: we_to_hazard | we_hazard_to_ppe\n"
            "\n"
            "- type: choose EXACTLY ONE of the following (string):\n"
            "  - add | remove | modify\n"
            "\n"
            "- target_from / target_to rules:\n"
            "  - add:    target_from=\"\" and target_to=\"NEW_KEY\"\n"
            "  - remove: target_from=\"EXISTING_KEY\" and target_to=\"\"\n"
            "  - modify: target_from=\"EXISTING_KEY\" and target_to=\"NEW_KEY\"\n"
            "\n"
            "- What 'label' proposals mean (kind=label):\n"
            "  - You are proposing to change the allowed label set itself.\n"
            "  - This includes adding/removing/modifying PPEItem keys (i.e., entirely new PPE types are allowed here).\n"
            "  - For kind=label, set scope.work_environment=\"\" and scope.hazard=\"\".\n"
            "\n"
            "- scope rules (only for kind=mapping):\n"
            "  - subject=we_to_hazard:\n"
            "    - scope.work_environment must be set (WorkEnvironment key)\n"
            "    - scope.hazard must be \"\"\n"
            "    - hazard key is expressed by target_from/target_to\n"
            "\n"
            "  - subject=we_hazard_to_ppe:\n"
            "    - scope.work_environment must be set (WorkEnvironment key)\n"
            "    - scope.hazard must be set (HazardFactor key)\n"
            "    - PPE key is expressed by target_from/target_to\n"
            "\n"
            "- scope rules (for kind=label):\n"
            "  - scope.work_environment=\"\" and scope.hazard=\"\"\n"
            "\n"
            "- proposal: short, evidence-based reason\n"
            "\n"
            "- IMPORTANT consistency rules:\n"
            "  - If you propose adding a new label (kind=label, type=add), also add at least one mapping proposal\n"
            "    (kind=mapping) that connects the new label so it is not isolated.\n"
            "  - If your mapping proposal uses a hazard/ppe key that does not exist in the current label set,\n"
            "    also include the corresponding kind=label add proposal for that hazard/ppe key.\n"
            "  - If the hazard/ppe key already exists in the current label set, do NOT write a kind=label proposal for it;\n"
            "    write only the required kind=mapping proposal(s).\n"
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
            "D) If mapping/labels seem insufficient, write proposals (or keep proposals=[] if no change).\n"
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
            "      \"proposals\": [<ProposalOut>, ...]\n"
            "    },\n"
            "    \"hazard\": {\n"
            "      \"hazards\": [<HazardFactor>, ...],\n"
            "      \"reason\": <string>,\n"
            "      \"proposals\": [<ProposalOut>, ...]\n"
            "    },\n"
            "    \"compliance\": {\n"
            "      \"required_ppe\": [<PPEItem>, ...],\n"
            "      \"reason\": <string>,\n"
            "      \"proposals\": [<ProposalOut>, ...]\n"
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
            "- proposals MUST be present for work_environment / hazard / compliance.\n"
            "  - If no change is needed, set proposals to an empty list: []\n"
        )
