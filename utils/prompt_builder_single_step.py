from __future__ import annotations

import json
from enum import Enum
from typing import Any, Type, TypeVar

from params.prompt_params import StageEnum, PromptPack
from params.structure import WorkEnvironment, HazardFactor, PPEItem
from params.ppe_mapping import WORK_ENV_TO_HAZARD_PPE
from pydantic import BaseModel
from params.common_prompt import CommonPrompt

E = TypeVar("E", bound=Enum)

def enum_keys(enum_cls: Type[E]):
    return [e.name for e in enum_cls]

class SingleStepPromptBuilder:
    # 공통 제약 사항
    STRICT_CONSTRAINTS = CommonPrompt.STRICT_CONSTRAINTS
    VISIBILITY_RULE = CommonPrompt.VISIBILITY_RULE
    SCHEMA_PROPOSAL_RULE = CommonPrompt.SCHEMA_PROPOSAL_RULE
    STAGE_GOALS = CommonPrompt.STAGE_GOALS
    PROPOSAL_POLICY_TEXT = CommonPrompt.PROPOSAL_POLICY_TEXT

    # =========================================================================
    # [METHODS]
    # =========================================================================

    def __init__(self):
        self.work_env_keys = enum_keys(WorkEnvironment)
        self.hazard_keys = enum_keys(HazardFactor)
        self.ppe_keys = enum_keys(PPEItem)

        self.proposal_stages = {
            StageEnum.WORK_ENVIRONMENT,
            StageEnum.HAZARD,
            StageEnum.COMPLIANCE,
        }

    # -----------------
    # public
    # -----------------
    def build(self, stage: StageEnum, state: dict[str, Any], text_format: BaseModel) -> PromptPack:
        return PromptPack(
            system=self.system_prompt(),
            user=self.user_prompt(stage, state, text_format),
        )

    # -----------------
    # system
    # -----------------
    def system_prompt(self) -> str:
        return CommonPrompt.SYSTEM_PROMPT + "This is SINGLE-AGENT PER STAGE labeling: one call per stage.\n"

    # -----------------
    # user
    # -----------------
    def get_text_format_schema(self, text_format: BaseModel):
        sc = text_format.model_json_schema()
        properties = sc.get("properties", {})
        clean = {fn: {k: v for k, v in fi.items() if k != "title"} for fn, fi in properties.items()}
        return json.dumps(clean, indent=2, ensure_ascii=False)

    def user_prompt(self, stage: StageEnum, state: dict[str, Any], text_format: BaseModel) -> str:
        compact = self.compact_state_for_stage(stage, state)
        state_json = json.dumps(compact, ensure_ascii=False)
        text_format_info = self.get_text_format_schema(text_format)

        parts = [
            self.constraint_block(stage, state, text_format_info),
            self.self_critique_block(),
            self.input_block(state_json),
            self.task_block(stage, state),
        ]
        return "\n".join([p for p in parts if p])

    def input_block(self, state_json: str) -> str:
        return "[Input]\n" f"State (previous outputs): {state_json}\n"

    # -----------------
    # constraints
    # -----------------
    def constraint_block(self, stage: StageEnum, state: dict[str, Any], text_format_info: str) -> str:
        parts = [
            "[Constraint]",
            *self.STRICT_CONSTRAINTS,
            f"6) FORMATTING: Follow the JSON schema below exactily: \n{text_format_info}",
            "Allowed labels:",
            self.allowed_labels_line(stage),
        ]

        hint = self.get_stage_mapping(stage, state)
        if hint:
            parts.append(f"\n{hint}")

        if stage in self.proposal_stages:
            parts.append("")
            parts.append(self.PROPOSAL_POLICY_TEXT)

        return "\n".join(parts) + "\n"

    def allowed_labels_line(self, stage: StageEnum) -> str:
        if stage == StageEnum.WORK_ENVIRONMENT: return f"- WorkEnvironment: {self.work_env_keys}"
        if stage == StageEnum.HAZARD: return f"- HazardFactor: {self.hazard_keys}"
        if stage == StageEnum.COMPLIANCE: return f"- PPEItem: {self.ppe_keys}"
        if stage in (StageEnum.WEARING, StageEnum.IMPROPER_WEARING):
            return f"- PPEItem: {self.ppe_keys}\n- Boolean: true/false"
        raise ValueError(f"unknown stage: {stage}")

    # -----------------
    # self-critique block (Simulates Rebutter/Judge)
    # -----------------
    def self_critique_block(self) -> str:
        return (
            "[Internal procedure]\n"
            "You must internally perform these steps before finalizing your JSON:\n"
            "A) Observation: Identify visible elements strictly based on the image.\n"
            "B) Self-Correction: Challenge your initial thoughts. Are you hallucinating invisible parts? Are you blindly following the mapping table without visual evidence?\n"
            "C) Final Decision: Determine the final labels that satisfy all constraints (Visual Evidence + Safety Rules).\n"
            "D) Proposal: If the schema is insufficient, populate the 'proposals' field.\n"
            "Only output the final JSON.\n"
        )

    # -----------------
    # stage hints (Same logic as Multi-Agent)
    # -----------------
    def get_stage_mapping(self, stage: StageEnum, state: dict[str, Any]) -> str:
        if stage == StageEnum.WORK_ENVIRONMENT:
            return ""

        if stage == StageEnum.HAZARD:
            we_name = state.get("work_environment")
            if not we_name:
                return "Context hint:\n- Previous work_environment is not decided yet."

            hazards_in_we = self.get_hazards_for_work_env(we_name)
            return (
                "[Ontology Results]\n"
                f"- Selected WorkEnvironment: {we_name}\n"
                f"- Defined HazardFactors for this environment: {hazards_in_we}\n"
                "- Note: You may suggest new factors or relationships if not listed above."
            )

        if stage == StageEnum.COMPLIANCE:
            we_name = state.get("work_environment")
            hazards = state.get("hazards", [])
            hazard_ppe_map = self.get_ppe_for_hazards(we_name, hazards)

            lines = ["[Ontology Results]"]
            lines.append(f"- Selected WorkEnvironment: {we_name}")
            lines.append(f"- Selected HazardFactors: {hazards}")

            if hazard_ppe_map:
                lines.append("- Defined PPE Mapping for selected hazards:")
                for h, ppes in hazard_ppe_map.items():
                    lines.append(f"  * {h}: {ppes}")
            else:
                lines.append("- No defined PPE mapping found for the selected context.")

            lines.append("\n- Note: You may choose from all PPEItem keys or suggest new ones if necessary.")
            return "\n".join(lines)

        return ""

    def get_hazards_for_work_env(self, we_name: str):
        try:
            we = WorkEnvironment[we_name]
            hps = WORK_ENV_TO_HAZARD_PPE.get(we, [])
            return list({hp.hazard.name for hp in hps})
        except Exception:
            return []

    def get_ppe_for_hazards(self, we_name: str, hazards: list[str]) -> dict[str, list[str]]:
        try:
            we = WorkEnvironment[we_name]
            hazard_set = set(hazards) if hazards else set()
            hps = WORK_ENV_TO_HAZARD_PPE.get(we, [])

            recommendation_map = {}
            for hp in hps:
                h_name = hp.hazard.name
                if h_name in hazard_set:
                    ppe_list = list(dict.fromkeys(p.name for p in hp.ppe))
                    recommendation_map[h_name] = ppe_list
            return recommendation_map
        except Exception:
            return {}


    def task_block(self, stage: StageEnum, state: dict[str, Any]) -> str:
        goal = self.STAGE_GOALS.get(stage, "")

        parts = ["[Task]", f"- MISSION: {goal}"]

        if stage in (StageEnum.WORK_ENVIRONMENT, StageEnum.HAZARD, StageEnum.COMPLIANCE):
            parts.append(self.SCHEMA_PROPOSAL_RULE)

        if stage in (StageEnum.WEARING, StageEnum.IMPROPER_WEARING):
            parts.append(self.VISIBILITY_RULE)

        # Stage specific additional instructions (Judge logic)
        if stage == StageEnum.COMPLIANCE:
            parts.append("- Rule: If a unique PPE is required for a specific hazard but missing in our mapping, list the current PPEs and then add a 'kind=mapping' proposal.")

        if stage == StageEnum.IMPROPER_WEARING:
            parts.append("- Note: Do NOT include items with worn=false in this list.")

        return "\n".join(parts)

    # -----------------
    # compact state
    # -----------------
    def compact_state_for_stage(self, stage: StageEnum, state: dict[str, Any]):
        base = {}

        # Single step has no stage_history usually, but if any, include it
        if history := state.get("stage_history"):
            base["current_stage_discussion"] = history

        we = state.get("work_environment")
        hazards = state.get("hazards")
        required_ppe = state.get("required_ppe")
        wearing = state.get("wearing")

        if stage == StageEnum.WORK_ENVIRONMENT:
            pass
        elif stage == StageEnum.HAZARD:
            base["work_environment"] = we
        elif stage == StageEnum.COMPLIANCE:
            base["work_environment"] = we
            base["hazards"] = hazards
        elif stage == StageEnum.WEARING:
            base.update({
                "work_environment": we,
                "hazards": hazards,
                "required_ppe": required_ppe
            })
        elif stage == StageEnum.IMPROPER_WEARING:
            base.update({
                "work_environment": we,
                "hazards": hazards,
                "required_ppe": required_ppe,
                "wearing": wearing
            })

        return base
