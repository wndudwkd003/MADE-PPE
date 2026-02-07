# utils/prompt_builder_single_step.py

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Type, TypeVar

from params.prompt_params import StageEnum, PromptPack
from params.structure import WorkEnvironment, HazardFactor, PPEItem
from params.ppe_mapping import WORK_ENV_TO_HAZARD_PPE

E = TypeVar("E", bound=Enum)


def enum_keys(enum_cls: Type[E]):
    return [e.name for e in enum_cls]


class SingleStepPromptBuilder:
    """
    - single-step(stage별 1회 호출) 전용 prompt builder
    - 출력은 stage별 최종 JSON schema(WorkEnvironmentOut/HazardOut/...)를 강제
    - 내부적으로 self-critique(= proposer/rebutter/judge 역할)를 한 번의 응답에서 수행하도록 유도
    """

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
    def build(self, stage: StageEnum, state: dict[str, Any]) -> PromptPack:
        return PromptPack(
            system=self.system_prompt(),
            user=self.user_prompt(stage, state),
        )
    
    # -----------------
    # system
    # -----------------
    def system_prompt(self) -> str:
        return (
            "You are an industrial safety labeling assistant for Personal Protective Equipment (PPE) compliance.\n"
            "You will be given an image and a state object from previous stages.\n"
            "Follow constraints and role instructions strictly.\n"
            "Keep reasons short and evidence-based.\n"
            "This is SINGLE-AGENT PER STAGE labeling: one call per stage.\n"
        )
    
    # -----------------
    # user
    # -----------------
    def user_prompt(self, stage: StageEnum, state: dict[str, Any]) -> str:
        compact = self.compact_state_for_stage(stage, state)
        state_json = json.dumps(compact, ensure_ascii=False)

        parts = [
            self.constraint_block(stage, state),
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
    def constraint_block(self, stage: StageEnum, state: dict[str, Any]) -> str:
        parts = [
            "[Constraint]",
            "1) Use ONLY the allowed labels listed below.",
            "2) If uncertain, choose the most plausible option based on visible evidence.",
            "3) Keep reasons short and evidence-based.",
            "",
            "Allowed labels:",
            self.allowed_labels_line(stage),
        ]

        hint = self.stage_hint_block(stage, state)
        if hint:
            parts += ["", hint]

        if stage in self.proposal_stages:
            parts += ["", self.proposal_policy_block()]

        return "\n".join(parts) + "\n"
    
    def allowed_labels_line(self, stage: StageEnum) -> str:
        if stage == StageEnum.WORK_ENVIRONMENT:
            return f"- WorkEnvironment: {self.work_env_keys}"
        if stage == StageEnum.HAZARD:
            return f"- HazardFactor: {self.hazard_keys}"
        if stage == StageEnum.COMPLIANCE:
            return f"- PPEItem: {self.ppe_keys}"
        if stage in (StageEnum.WEARING, StageEnum.IMPROPER_WEARING):
            return f"- PPEItem: {self.ppe_keys}\n- Boolean: true/false"
        raise ValueError(f"unknown stage: {stage}")
    
    def proposal_policy_block(self) -> str:
       return (
            "Proposal policy (for schema/mapping/label change):\n"
            "- proposal.flag:\n"
            "  - true  => you are proposing a change.\n"
            "  - false => no change needed.\n"
            "\n"
            "- proposal.type: choose ONE of the following (string):\n"
            "  - add | remove | modify\n"
            "\n"
            "- proposal.target_from:\n"
            "  - If type is modify or remove: MUST be an existing item in the current allowed label list.\n"
            "  - If type is add: use empty string \"\".\n"
            "\n"
            "- proposal.target_to:\n"
            "  - If type is add or modify: the desired new item/key.\n"
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
    def self_critique_block(self) -> str:
        return (
            "[Internal procedure]\n"
            "You must internally perform these steps before finalizing your JSON:\n"
            "A) Proposer: draft the best candidate label(s) using visible evidence.\n"
            "B) Rebutter: challenge your own draft (missing evidence? alternative interpretation?).\n"
            "C) Judge: finalize the answer that best satisfies constraints.\n"
            "D) If mapping/labels seem insufficient, use proposal fields (or keep them empty if no change).\n"
            "Only output the final JSON.\n"
        )
    
    # -----------------
    # stage hints (mapping)
    # -----------------
    def stage_hint_block(self, stage: StageEnum, state: dict[str, Any]) -> str:
        if stage == StageEnum.WORK_ENVIRONMENT:
            return ""

        if stage == StageEnum.HAZARD:
            we_name = state.get("work_environment")
            if not we_name:
                return "Context hint:\n- Previous work_environment is not decided yet."

            hazards_in_we = self._hazards_for_work_env(we_name)
            if hazards_in_we:
                return (
                    "Context hint:\n"
                    f"- Selected work_environment: {we_name}\n"
                    f"- Typical hazards for this work_environment: {hazards_in_we}\n"
                    f"- You can still choose from all HazardFactor keys listed above.\n"
                )
            return (
                "Context hint:\n"
                f"- Selected work_environment: {we_name}\n"
                "- No mapping hazards found for this work_environment.\n"
            )

        if stage == StageEnum.COMPLIANCE:
            we_name = state.get("work_environment")
            hazards = state.get("hazards")

            lines = ["Context hint:"]
            lines.append(f"- Selected work_environment: {we_name}" if we_name else "- Selected work_environment: (missing)")
            lines.append(f"- Selected hazards: {hazards}" if hazards else "- Selected hazards: (missing)")

            recommended = self._recommended_ppe(we_name, hazards)
            lines.append(
                f"- Recommended PPE given selected context: {recommended}"
                if recommended
                else "- Recommended PPE given selected context: (none from mapping)"
            )
            lines.append("- You can still choose from all PPEItem keys listed above.")
            return "\n".join(lines)

        return ""
    
    def _hazards_for_work_env(self, we_name: str):
        try:
            we = WorkEnvironment[we_name]
        except Exception:
            return []

        hps = WORK_ENV_TO_HAZARD_PPE.get(we, [])
        hazards = []
        for hp in hps:
            hazards.append(hp.hazard.name)

        out = []
        seen = set()
        for h in hazards:
            if h not in seen:
                seen.add(h)
                out.append(h)
        return out
    
    def _recommended_ppe(self, we_name: str | None, hazards: list[str] | None):
        if not we_name:
            return []
        try:
            we = WorkEnvironment[we_name]
        except Exception:
            return []

        hazard_set = set(hazards) if hazards else set()
        hps = WORK_ENV_TO_HAZARD_PPE.get(we, [])

        ppe_names = []
        for hp in hps:
            if (not hazard_set) or (hp.hazard.name in hazard_set):
                for p in hp.ppe:
                    ppe_names.append(p.name)

        out = []
        seen = set()
        for p in ppe_names:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out
    
    # -----------------
    # tasks
    # -----------------
    def task_block(self, stage: StageEnum, state: dict[str, Any]) -> str:
        if stage == StageEnum.WORK_ENVIRONMENT:
            return (
                "[Task]\n"
                "- Output WorkEnvironmentOut JSON.\n"
                "- Choose exactly 1 WorkEnvironment key.\n"
                "- Provide a short reason.\n"
                "- Fill proposal fields (even if no change; follow proposal policy).\n"
            )
        if stage == StageEnum.HAZARD:
            return (
                "[Task]\n"
                "- Output HazardOut JSON.\n"
                "- Select all applicable HazardFactor keys.\n"
                "- Provide a short reason.\n"
                "- Fill proposal fields (even if no change; follow proposal policy).\n"
            )
        if stage == StageEnum.COMPLIANCE:
            return (
                "[Task]\n"
                "- Output ComplianceOut JSON.\n"
                "- Decide required_ppe as FINAL PPE list.\n"
                "- Provide a short reason.\n"
                "- Fill proposal fields (even if no change; follow proposal policy).\n"
            )
        if stage == StageEnum.WEARING:
            return (
                "[Task]\n"
                "- Output WearingOut JSON.\n"
                "- Using state.required_ppe, decide worn=true/false for each PPE item.\n"
                "- Provide a short reason.\n"
            )
        if stage == StageEnum.IMPROPER_WEARING:
            return (
                "[Task]\n"
                "- Output ImproperWearingOut JSON.\n"
                "- Consider ONLY items in state.wearing where worn=true.\n"
                "- Output improper_wearing list ONLY for those items.\n"
                "- improper_wearing.worn=true means 'worn but incorrectly'; false means 'worn correctly'.\n"
                "- Provide a short reason.\n"
            )
        raise ValueError(f"unknown stage: {stage}")
    
    # -----------------
    # compact state
    # -----------------
    def compact_state_for_stage(self, stage: StageEnum, state: dict[str, Any]):
        base: dict[str, Any] = {}

        if stage == StageEnum.WORK_ENVIRONMENT:
            return base

        if stage == StageEnum.HAZARD:
            if "work_environment" in state:
                base["work_environment"] = state["work_environment"]
            return base

        if stage == StageEnum.COMPLIANCE:
            if "work_environment" in state:
                base["work_environment"] = state["work_environment"]
            if "hazards" in state:
                base["hazards"] = state["hazards"]
            return base

        if stage == StageEnum.WEARING:
            if "required_ppe" in state:
                base["required_ppe"] = state["required_ppe"]
            return base

        if stage == StageEnum.IMPROPER_WEARING:
            if "required_ppe" in state:
                base["required_ppe"] = state["required_ppe"]
            if "wearing" in state:
                base["wearing"] = state["wearing"]
            return base

        return base