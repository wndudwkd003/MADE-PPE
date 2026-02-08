from __future__ import annotations

import json
from enum import Enum
from typing import Any, Type, TypeVar

from pyparsing import C

from params.prompt_params import RoleEnum, StageEnum, PromptPack
from params.structure import WorkEnvironment, HazardFactor, PPEItem
from params.ppe_mapping import WORK_ENV_TO_HAZARD_PPE
from pydantic import BaseModel
from params.common_prompt import CommonPrompt

E = TypeVar("E", bound=Enum)

def enum_keys(enum_cls: Type[E]):
    return [e.name for e in enum_cls]



class PromptBuilder:

    # 공통 제약 사항
    STRICT_CONSTRAINTS = CommonPrompt.STRICT_CONSTRAINTS
    VISIBILITY_RULE = CommonPrompt.VISIBILITY_RULE
    SCHEMA_PROPOSAL_RULE = CommonPrompt.SCHEMA_PROPOSAL_RULE
    STAGE_GOALS = CommonPrompt.STAGE_GOALS
    PROPOSAL_POLICY_TEXT = CommonPrompt.PROPOSAL_POLICY_TEXT

    # --

    REBUTTER_COMMON_TASK = (
        "REBUTTER:\n"
        "- Start your text with 'AGREE' or 'DISAGREE'.\n"
        "- You must reference the proposer's claim if it exists in the state.\n"
        "- If DISAGREE, you must provide corrected enum keys and brief evidence.\n"
        "- Check for hallucinations: If the proposer claims status for a body part not visible in the image, you must DISAGREE.\n"
    )

    JUDGE_COMMON_GUIDELINE = (
        "- FINAL DECISION: Resolve the debate. The final labels in the main fields MUST exist in the current 'Allowed labels'.\n"
        "- NO ASSUMPTIONS: Do not allow 'guessing' for missing part of the body (hands, feet, etc.).\n"
        "- STRICT SCHEMA: Never output a non-existent key in the label fields. This will cause a system error.\n"
        "- PROPOSAL CAPTURE: If a Proposer or Rebutter suggested a valid new WorkEnvironment, PPE, Hazard, or Mapping that isn't in our system yet, "
        "please review its appropriateness and consider adding it to the 'proposals' list field.\n"
    )

    # --

    def __init__(self):
        self.work_env_keys = enum_keys(WorkEnvironment)
        self.hazard_keys = enum_keys(HazardFactor)
        self.ppe_keys = enum_keys(PPEItem)
        self.proposal_stages = {StageEnum.WORK_ENVIRONMENT, StageEnum.HAZARD, StageEnum.COMPLIANCE}

    def build(self, stage: StageEnum, role: RoleEnum, round_idx: int, state: dict[str, Any], text_format: BaseModel):
        return PromptPack(
            system=self.system_prompt(role),
            user=self.user_prompt(stage, role, round_idx, state, text_format),
        )

    def system_prompt(self, role: RoleEnum):
        return CommonPrompt.SYSTEM_PROMPT + f"Role: {role.name}.\n"

    def get_text_format_schema(self, text_format: BaseModel):
        sc = text_format.model_json_schema()
        properties = sc.get("properties", {})
        clean = {fn: {k: v for k, v in fi.items() if k != "title"} for fn, fi in properties.items()}
        return json.dumps(clean, indent=2, ensure_ascii=False)

    def user_prompt(self, stage: StageEnum, role: RoleEnum, round_idx: int, state: dict[str, Any], text_format: BaseModel):
        compact = self.compact_state_for_stage(stage, state)
        state_json = json.dumps(compact, ensure_ascii=False)
        text_format_info = self.get_text_format_schema(text_format)

        parts = [
            self.constraint_block(stage, role, state, text_format_info),
            self.role_block(role, round_idx),
            self.input_block(state_json),
            self.task_block(stage, role, round_idx),
        ]
        return "\n".join(parts)

    def input_block(self, state_json: str):
        return "[Input]\n" f"State (previous outputs): {state_json}\n"

    def constraint_block(self, stage: StageEnum, role: RoleEnum, state: dict[str, Any], text_format_info: str):
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

        if role == RoleEnum.JUDGE and stage in self.proposal_stages:
            parts.append("")
            parts.append(self.PROPOSAL_POLICY_TEXT)

        return "\n".join([p for p in parts if p]) + "\n"

    def allowed_labels_line(self, stage: StageEnum):
        if stage == StageEnum.WORK_ENVIRONMENT: return f"- WorkEnvironment: {self.work_env_keys}"
        if stage == StageEnum.HAZARD: return f"- HazardFactor: {self.hazard_keys}"
        if stage == StageEnum.COMPLIANCE: return f"- PPEItem: {self.ppe_keys}"
        if stage in (StageEnum.WEARING, StageEnum.IMPROPER_WEARING):
            return f"- PPEItem: {self.ppe_keys}\n- Boolean: true/false"
        return ""

    # --- Data Retrieval Logic ---
    def hazard_to_ppe_map_for_work_env(self, we: WorkEnvironment):
        hps = WORK_ENV_TO_HAZARD_PPE.get(we, [])
        out = {}
        for hp in hps:
            out[hp.hazard.name] = [p.name for p in hp.ppe]
        return out

    def get_stage_mapping(self, stage: StageEnum, state: dict[str, Any]):
        we_name = state.get("work_environment")
        if not we_name:
            return ""

        if stage == StageEnum.HAZARD:
            hazards_in_we = self.get_hazards_for_work_env(we_name)
            return (
                "[Ontology Results]\n"
                f"- Selected WorkEnvironment: {we_name}\n"
                f"- Defined HazardFactors for this environment: {hazards_in_we}\n"
                "- Note: You may suggest new factors or relationships if not listed above."
            )

        if stage == StageEnum.COMPLIANCE:
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

    def role_block(self, role: RoleEnum, round_idx: int):
        if role == RoleEnum.PROPOSER:
            if round_idx == 0:
                return (
                    "[Role: Initial Proposer]\n"
                    "- Provide the most accurate prediction based on the image.\n"
                    "- The description of the situation related to the label must also be included.\n"
                    "- If current labels or mapping rules are insufficient for this case, suggest schema changes.\n"
                    "- It is important to provide answers that match the current step early on.\n"
                )
            return (
                "[Role: Advanced Proposal]\n"
                "- If the rebutter AGREED, find additional/stronger visual evidence to solidify the claim.\n"
                "- If the rebutter DISAGREED, adjust your prediction and justification by incorporating the feedback.\n"
                "- The initial proposer may make mistakes. You can provide a more developed response.\n"
            )

        if role == RoleEnum.REBUTTER:
            return (
                "[Role: Critical Validator]\n"
                "- Evaluate if the proposer's labels align with the image AND the current rules.\n"
                "- You must carefully verify the discrepancy between the proposer's statement and the actual image.\n"
                "- You must prevent the proposer from giving an incorrect answer. Visual evidence is required.\n"
                "- Judge if the 'evidence' provided is logically sound and clearly visible.\n"
                "- Start with AGREE or DISAGREE.\n"
            )

        if role == RoleEnum.JUDGE:
            return (
                "[Role: Final Authority]\n"
                "- Choose labels based on current active rules and image evidence.\n"
                "- Also, you must find evidence based on the responses from the proponent and the opponent.\n"
                "- If labels/mappings are fundamentally ill-suited for this image, include a 'proposals' list for schema improvement.\n"
            )
        return ""

    def task_block(self, stage: StageEnum, role: RoleEnum, round_idx: int):
        goal = self.STAGE_GOALS.get(stage, "")

        if role == RoleEnum.PROPOSER:
            prefix = "PROPOSER (Initial Proposal):" if round_idx == 0 else "PROPOSER (Reinforcement/Adjustment):"
            parts = [f"[Task]\n{prefix}", f"- MISSION: {goal}"]

            if round_idx == 0:
                parts += [
                    self.VISIBILITY_RULE,
                    self.SCHEMA_PROPOSAL_RULE,
                    "- FORMAT: Use only current allowed labels for the JSON prediction, but advocate for changes in your text."
                ]
            else:
                parts.append("- Review the Rebutter's feedback. If they disagree because of a missing label in our system, solidify the proposal for that new label/mapping in your response.")
            return "\n".join(parts)

        if role == RoleEnum.REBUTTER:
            extra = ""
            if stage == StageEnum.WORK_ENVIRONMENT: extra = "- If DISAGREE, give exactly 1 corrected WorkEnvironment key.\n"
            elif stage == StageEnum.HAZARD: extra = "- If DISAGREE, list corrected HazardFactor keys.\n"
            elif stage == StageEnum.COMPLIANCE: extra = "- If DISAGREE, provide corrected FINAL PPEItem keys.\n"
            elif stage == StageEnum.WEARING: extra = "- If DISAGREE, provide corrected true/false per PPE item.\n"
            elif stage == StageEnum.IMPROPER_WEARING: extra = "- If DISAGREE, provide corrected true/false per PPE item.\n"

            return f"[Task]\n{self.REBUTTER_COMMON_TASK}{extra}"

        if role == RoleEnum.JUDGE:
            action = ""
            if stage == StageEnum.WORK_ENVIRONMENT:
                action = "- Action: Choose exactly 1 CURRENT WorkEnvironment key.\n- Proposal: If the image shows an environment we don't have, select the closest current one AND add a 'kind=label' proposal."
            elif stage == StageEnum.HAZARD:
                action = "- Action: Select ALL applicable HazardFactor keys from the CURRENT list.\n- Note: Do not select mapped hazards if they aren't visible. Do not invent new hazard names in the HazardFactor field."
            elif stage == StageEnum.COMPLIANCE:
                action = "- Action: Decide the FINAL PPEItem list using only CURRENT keys.\n- Rule: If a unique PPE is required for a specific hazard but missing in our mapping, list the current PPEs and then add a 'kind=mapping' proposal to include the new one."
            elif stage == StageEnum.WEARING:
                action = f"- Output: WearingOut (JSON).\n- Action: Using state.required_ppe, decide worn=true/false for each PPE item.\n- Note: {self.VISIBILITY_RULE.split(': ')[1]}"
            elif stage == StageEnum.IMPROPER_WEARING:
                action = "- Output: ImproperWearingOut (JSON).\n- Constraint: Consider ONLY items in state.wearing where worn=true.\n- Action: Set improper_wearing.worn=true ONLY if the item is clearly visible and worn incorrectly. Else false.\n- Note: Do NOT include items with worn=false in this list."

            return f"[Task]\n{self.JUDGE_COMMON_GUIDELINE}{action}"

        raise ValueError(f"unknown stage/role: {stage}/{role}")

    def compact_state_for_stage(self, stage: StageEnum, state: dict[str, Any]):
        base = {}

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
