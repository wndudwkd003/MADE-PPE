from __future__ import annotations

import json
from enum import Enum
from pydoc import text
from typing import Any, Type, TypeVar

from params.prompt_params import RoleEnum, StageEnum, PromptPack
from params.structure import WorkEnvironment, HazardFactor, PPEItem
from params.ppe_mapping import WORK_ENV_TO_HAZARD_PPE
from pydantic import BaseModel

E = TypeVar("E", bound=Enum)


def enum_keys(enum_cls: Type[E]):
    return [e.name for e in enum_cls]


class PromptBuilder:
    def __init__(self):
        self.work_env_keys = enum_keys(WorkEnvironment)
        self.hazard_keys = enum_keys(HazardFactor)
        self.ppe_keys = enum_keys(PPEItem)

        self.proposal_stages = {
            StageEnum.WORK_ENVIRONMENT,
            StageEnum.HAZARD,
            StageEnum.COMPLIANCE,
        }

    def build(
        self,
        stage: StageEnum,
        role: RoleEnum,
        round_idx: int,
        state: dict[str, Any],
        text_format: BaseModel,
    ):
        return PromptPack(
            system=self.system_prompt(role),
            user=self.user_prompt(stage, role, round_idx, state, text_format),
        )

    # -----------------
    # -----------------

    def system_prompt(
        self,
        role: RoleEnum
    ):
        return (
            "You are an industrial safety labeling assistant for Personal Protective Equipment (PPE) compliance.\n"
            "You will be given an image and a state object from previous stages.\n"
            "Follow constraints and role instructions strictly.\n"
            "Keep reasons short and evidence-based.\n"
            f"Role: {role.name}.\n"
        )

    def get_text_format_schema(
        self,
        text_format: BaseModel,
    ):
        sc = text_format.model_json_schema()
        properties = sc.get("properties", {})

        clean = {}
        for fn, fi in properties.items():
            info = dict(fi)
            info.pop("title", None)
            clean[fn] = info

        text_format_info = json.dumps(
            clean,
            indent=2, ensure_ascii=False
        )

        return text_format_info


    def user_prompt(
        self,
        stage: StageEnum, # we, hazard, compliance, wearing, improper_wearing
        role: RoleEnum,
        round_idx: int,
        state: dict[str, Any],
        text_format: BaseModel,
    ):
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


    def constraint_block(
        self,
        stage: StageEnum,
        role: RoleEnum,
        state: dict[str, Any],
        text_format_info: str,
    ):
        parts = [
            "[Constraint]",
            "1) STRICT LABELING: For final JSON output fields (WorkEnvironment, HazardFactor, PPEItem), use ONLY the allowed labels listed below. Do NOT invent new keys for these fields.",
            "2) EVIDENCE OVER MAPPING: Mapping tables are reference guides. Select only what is visually supported. If a mapped item is missing in the image, do NOT include it.",
            "3) SCHEMA PROPOSALS: Proposers and Judges can suggest new labels or mappings, but these MUST be placed in the justification text or the 'proposals' list, never as a label in the main fields.",
            "4) NO HALLUCINATION: If a body part (e.g., hands, feet) is not visible in the image, do NOT assume or invent PPE status. Never make up facts for hidden or out-of-frame areas.",
            "5) UPPER BOUND ONLY: The defined labels and mapping structures are the MAXIMUM UPPER BOUND, not a minimum requirement. You are NOT obligated to fill every field or select all mapped items. Label ONLY what is necessary based on the specific evidence in THIS image.",
            f"6) FORMATTING: Follow the JSON schema below exactily: \n{text_format_info}",
            "Allowed labels:",
            self.allowed_labels_line(stage),
        ]

        hint = self.get_stage_mapping(stage, state)
        if hint:
            parts.append(f"\n{hint}")

        if role == RoleEnum.JUDGE and stage in self.proposal_stages:
            parts += ["", self.proposal_policy_block()]

        return "\n".join([p for p in parts if p]) + "\n"

    def allowed_labels_line(self, stage: StageEnum):
        if stage == StageEnum.WORK_ENVIRONMENT:
            return f"- WorkEnvironment: {self.work_env_keys}"
        if stage == StageEnum.HAZARD:
            return f"- HazardFactor: {self.hazard_keys}"
        if stage == StageEnum.COMPLIANCE:
            return f"- PPEItem: {self.ppe_keys}"
        if stage in (StageEnum.WEARING, StageEnum.IMPROPER_WEARING):
            return f"- PPEItem: {self.ppe_keys}\n- Boolean: true/false"

    def proposal_policy_block(self):
        return (
            "Proposal policy (JUDGE only, for label-set / mapping-table changes):\n"
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

    def hazard_to_ppe_map_for_work_env(self, we: WorkEnvironment):
        hps = WORK_ENV_TO_HAZARD_PPE.get(we, [])
        out = {}
        for hp in hps:
            out[hp.hazard.name] = [p.name for p in hp.ppe]
        return out

    def mapping_subset_lines(self, we_name: str | None, hazards: list[str] | None):
        if not we_name:
            return []
        if not hazards:
            return []
        try:
            we = WorkEnvironment[we_name]
        except Exception:
            return []

        hazard_to_ppe = self.hazard_to_ppe_map_for_work_env(we)

        lines = []
        for h in hazards:
            if h in hazard_to_ppe:
                lines.append(f"- {h} -> {hazard_to_ppe[h]}")

        if not lines:
            return []
        return ["Mapping context (selected hazards only):"] + lines


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
            # 위험요소: [PPE들] 매핑 데이터 가져오기
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
        except Exception:
            return []

        hps = WORK_ENV_TO_HAZARD_PPE.get(we, [])

        seen_hazards = {hp.hazard.name for hp in hps}

        return list(seen_hazards)

    def get_ppe_for_hazards(
        self,
        we_name: str,
        hazards: list[str]
    ) -> dict[str, list[str]]:
        try:
            we = WorkEnvironment[we_name]
        except Exception:
            return {}

        hazard_set = set(hazards) if hazards else set()
        hps = WORK_ENV_TO_HAZARD_PPE.get(we, [])

        recommendation_map = {}

        for hp in hps:
            h_name = hp.hazard.name
            if h_name in hazard_set:
                ppe_list = list(dict.fromkeys(p.name for p in hp.ppe))
                recommendation_map[h_name] = ppe_list

        return recommendation_map


    def role_block(
        self,
        role: RoleEnum,
        round_idx: int
    ):
        if role == RoleEnum.PROPOSER:
            if round_idx == 0:
                return (
                    "[Role: Initial Proposer]\n"
                    "- Provide the most accurate prediction based on the image.\n"
                    "- The description of the situation related to the label must also be included.\n"
                    "- If current labels or mapping rules are insufficient for this case, suggest schema changes.\n"
                    "- It is important to provide answers that match the current step early on.\n"
                )
            else:
                return (
                    "[Role: Advanced Proposal, Reinforcer/Adapter]\n"
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

    def task_block(self, stage: StageEnum, role: RoleEnum, round_idx: int):
        if role == RoleEnum.PROPOSER:
            return "[Task]\n" + self.textout_task_proposer(stage, round_idx)
        if role == RoleEnum.REBUTTER:
            return "[Task]\n" + self.textout_task_rebutter(stage)
        if role == RoleEnum.JUDGE:
            return self.task_judge(stage)


    def textout_task_proposer(self, stage: StageEnum, round_idx: int):
        stage_goals = {
            StageEnum.WORK_ENVIRONMENT: "Choose 1 WorkEnvironment key.",
            StageEnum.HAZARD: "List all applicable HazardFactor keys.",
            StageEnum.COMPLIANCE: "Decide the FINAL required PPE list (PPEItem keys).",
            StageEnum.WEARING: "Decide worn=true/false ONLY for PPE visible on body parts shown in the image.",
            StageEnum.IMPROPER_WEARING: "For visible and worn=true items, decide if they are worn incorrectly."
        }
        goal = stage_goals.get(stage, "")

        if round_idx == 0:
            return (
                f"PROPOSER (Initial Proposal):\n"
                f"- MISSION: {goal}\n"
                f"- VISIBILITY RULE: If hands or feet are out of frame or obscured, state 'not visible' in reasoning and do NOT set worn=true based on a guess.\n"
                f"- SCHEMA PROPOSAL: If you see a new WorkEnvironment, Hazard, or PPE that fits this image but is NOT in the 'Allowed labels', "
                f"you MUST describe it in your reasoning. Suggest it as a new label or a new mapping rule.\n"
                f"- FORMAT: Use only current allowed labels for the JSON prediction, but advocate for changes in your text."
            )
        else:
            return (
                f"PROPOSER (Reinforcement/Adjustment):\n"
                f"- MISSION: {goal}\n"
                f"- Review the Rebutter's feedback. If they disagree because of a missing label in our system, "
                f"solidify the proposal for that new label/mapping in your response."
            )

    def textout_task_rebutter(self, stage: StageEnum):
        common = (
            "REBUTTER:\n"
            "- Start your text with 'AGREE' or 'DISAGREE'.\n"
            "- You must reference the proposer's claim if it exists in the state.\n"
            "- If DISAGREE, you must provide corrected enum keys and brief evidence.\n"
            "- Check for hallucinations: If the proposer claims status for a body part not visible in the image, you must DISAGREE.\n"
        )

        if stage == StageEnum.WORK_ENVIRONMENT:
            return common + "- If DISAGREE, give exactly 1 corrected WorkEnvironment key.\n"
        if stage == StageEnum.HAZARD:
            return common + "- If DISAGREE, list corrected HazardFactor keys.\n"
        if stage == StageEnum.COMPLIANCE:
            return common + "- If DISAGREE, provide corrected FINAL PPEItem keys.\n"
        if stage == StageEnum.WEARING:
            return common + "- If DISAGREE, provide corrected true/false per PPE item.\n"
        if stage == StageEnum.IMPROPER_WEARING:
            return common + "- If DISAGREE, provide corrected true/false per PPE item.\n"

        raise ValueError(f"unknown stage: {stage}")

    def task_judge(self, stage: StageEnum):
        common_guideline = (
            "- FINAL DECISION: Resolve the debate. The final labels in the main fields MUST exist in the current 'Allowed labels'.\n"
            "- NO ASSUMPTIONS: If a specific part of the body (hands, feet, etc.) is not captured in the image, reject any claims about PPE for those parts. Do not allow 'guessing' for missing extremities.\n"
            "- STRICT SCHEMA: Never output a non-existent key in the label fields. This will cause a system error.\n"
            "- PROPOSAL CAPTURE: If a Proposer or Rebutter suggested a valid new WorkEnvironment, PPE, Hazard, or Mapping that isn't in our system yet, "
            "you MUST record this in the 'proposals' list field. This is the ONLY way to introduce new elements to the system.\n"
        )

        if stage == StageEnum.WORK_ENVIRONMENT:
            return (
                f"[Task]\n{common_guideline}"
                f"- Action: Choose exactly 1 CURRENT WorkEnvironment key.\n"
                f"- Proposal: If the image shows an environment we don't have, select the closest current one AND add a 'kind=label' proposal."
            )

        if stage == StageEnum.HAZARD:
            return (
                f"[Task]\n{common_guideline}"
                f"- Action: Select ALL applicable HazardFactor keys from the CURRENT list.\n"
                f"- Note: Do not select mapped hazards if they aren't visible. Do not invent new hazard names in the HazardFactor field."
            )

        if stage == StageEnum.COMPLIANCE:
            return (
                f"[Task]\n{common_guideline}"
                f"- Action: Decide the FINAL PPEItem list using only CURRENT keys.\n"
                f"- Rule: If a unique PPE is required for a specific hazard but missing in our mapping, "
                f"list the current PPEs and then add a 'kind=mapping' proposal to include the new one."
            )

        if stage == StageEnum.WEARING:
            return (
                f"[Task]\n{common_guideline}"
                f"- Output: WearingOut (JSON).\n"
                f"- Action: Using state.required_ppe, decide worn=true/false for each PPE item.\n"
                f"- Note: If the body part is missing from the image, you must explain that it cannot be determined and avoid setting worn=true based on a guess."
            )

        if stage == StageEnum.IMPROPER_WEARING:
            return (
                f"[Task]\n{common_guideline}"
                f"- Output: ImproperWearingOut (JSON).\n"
                f"- Constraint: Consider ONLY items in state.wearing where worn=true.\n"
                f"- Action: Set improper_wearing.worn=true ONLY if the item is clearly visible and worn incorrectly. Else false.\n"
                f"- Note: Do NOT include items with worn=false in this list.\n"
            )

        raise ValueError(f"unknown stage: {stage}")


    def compact_state_for_stage(
        self,
        stage: StageEnum,
        state: dict[str, Any]
    ):
        base = {}

        # 1. 현재 스테이지의 논의 이력은 항상 최우선으로 포함
        if history := state.get("stage_history"):
            base["current_stage_discussion"] = history

        # 2. 필요한 변수들 미리 확보 (안전한 추출)
        we = state.get("work_environment")
        hazards = state.get("hazards")
        required_ppe = state.get("required_ppe")
        wearing = state.get("wearing")

        # 3. 스테이지별로 '꼭 필요한 만큼만' 누적해서 담기
        if stage == StageEnum.WORK_ENVIRONMENT:
            pass # 추가 정보 없음

        elif stage == StageEnum.HAZARD:
            base["work_environment"] = we

        elif stage == StageEnum.COMPLIANCE:
            base["work_environment"] = we
            base["hazards"] = hazards

        elif stage == StageEnum.WEARING:
            # 이 단계부터는 결정된 PPE 목록이 핵심입니다.
            base.update({
                "work_environment": we,
                "hazards": hazards,
                "required_ppe": required_ppe
            })

        elif stage == StageEnum.IMPROPER_WEARING:
            # 이전 단계의 착용 여부(wearing)가 반드시 필요합니다.
            base.update({
                "work_environment": we,
                "hazards": hazards,
                "required_ppe": required_ppe,
                "wearing": wearing
            })

        return base
