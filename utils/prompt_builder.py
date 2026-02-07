# utils/prompt_builder.py

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Type, TypeVar

from params.prompt_params import RoleEnum, StageEnum, PromptPack
from params.structure import WorkEnvironment, HazardFactor, PPEItem
from params.ppe_mapping import WORK_ENV_TO_HAZARD_PPE

E = TypeVar("E", bound=Enum)


def enum_keys(enum_cls: Type[E]):
    return [e.name for e in enum_cls]


class PromptBuilder:
    """
    - state는 절대 "수정/주입/매핑"하지 않는다.
    - state를 읽어서 프롬프트에 필요한 요약만 출력한다.
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
    def build(self, stage: StageEnum, role: RoleEnum, round_idx: int, state: dict[str, Any]):
        return PromptPack(
            system=self.system_prompt(role),
            user=self.user_prompt(stage, role, round_idx, state),
        )

    # -----------------
    # system
    # -----------------
    def system_prompt(self, role: RoleEnum):
        return (
            "You are an industrial safety labeling assistant for Personal Protective Equipment (PPE) compliance.\n"
            "You will be given an image and a state object from previous stages.\n"
            "Follow constraints and role instructions strictly.\n"
            "Keep reasons short and evidence-based.\n"
            f"Role: {role.name}.\n"
        )

    # -----------------
    # user
    # -----------------
    def user_prompt(self, stage: StageEnum, role: RoleEnum, round_idx: int, state: dict[str, Any]):
        compact = self.compact_state_for_stage(stage, role, round_idx, state)
        state_json = json.dumps(compact, ensure_ascii=False)

        parts = [
            self.constraint_block(stage, role, state),
            self.role_block(role),
            self.input_block(state_json),
            self.task_block(stage, role),
        ]
        return "\n".join(parts)

    def input_block(self, state_json: str):
        return "[Input]\n" f"State (previous outputs): {state_json}\n"

    # -----------------
    # constraint
    # -----------------
    def constraint_block(self, stage: StageEnum, role: RoleEnum, state: dict[str, Any]):
        parts = [
            "[Constraint]",
            "1) Use ONLY the allowed labels listed below.",
            "2) If uncertain, choose the most plausible option based on visible evidence.",
            "3) Keep reasons short and evidence-based.",
            self.output_policy_line(role),
            "",
            "Allowed labels:",
            self.allowed_labels_line(stage),
        ]

        hint = self.stage_hint_block(stage, state)
        if hint:
            parts += ["", hint]

        if role == RoleEnum.JUDGE and stage in self.proposal_stages:
            parts += ["", self.proposal_policy_block()]

        return "\n".join([p for p in parts if p]) + "\n"

    def output_policy_line(self, role: RoleEnum):
        if role in (RoleEnum.PROPOSER, RoleEnum.REBUTTER):
            return '4) Output must be a single JSON object matching TextOut: {"text": "..."} (no other keys).'
        return "4) Output must be a single JSON object matching the provided stage JSON Schema (no extra keys)."

    def allowed_labels_line(self, stage: StageEnum):
        if stage == StageEnum.WORK_ENVIRONMENT:
            return f"- WorkEnvironment: {self.work_env_keys}"
        if stage == StageEnum.HAZARD:
            return f"- HazardFactor: {self.hazard_keys}"
        if stage == StageEnum.COMPLIANCE:
            return f"- PPEItem: {self.ppe_keys}"
        if stage in (StageEnum.WEARING, StageEnum.IMPROPER_WEARING):
            return f"- PPEItem: {self.ppe_keys}\n- Boolean: true/false"
        raise ValueError(f"unknown stage: {stage}")

    def proposal_policy_block(self):
        return (
            "Proposal policy (JUDGE only, for label-set / mapping-table changes):\n"
            "\n"
            "- You must output proposals as a list field named: proposals\n"
            "- If no changes are needed, set proposals to an empty list: []\n"
            "\n"
            "Each item in proposals must match ProposalOut:\n"
            "- flag: must be true for every proposal item in the list\n"
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
            "- IMPORTANT:\n"
            "  - If you propose adding a new label (kind=label, type=add), you should also include\n"
            "    at least one mapping proposal (kind=mapping) that connects the new label so it is not isolated.\n"
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


    # -----------------
    # stage hint
    # -----------------
    def stage_hint_block(self, stage: StageEnum, state: dict[str, Any]):
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

            mapping_lines = self.mapping_subset_lines(we_name, hazards)

            if mapping_lines:
                lines.append("")
                lines.extend(mapping_lines)

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
    # role / task
    # -----------------
    def role_block(self, role: RoleEnum):
        if role == RoleEnum.PROPOSER:
            return (
                "[Role]\n"
                "- Produce the best initial prediction.\n"
                "- Mention enum keys exactly when referring to labels.\n"
            )
        if role == RoleEnum.REBUTTER:
            return (
                "[Role]\n"
                "- Critique the proposer based on the given state and image evidence.\n"
                "- Explicitly start with AGREE or DISAGREE.\n"
                "- If DISAGREE, provide corrected enum keys.\n"
                "- Mention enum keys exactly.\n"
            )
        if role == RoleEnum.JUDGE:
            return (
                "[Role]\n"
                "- Decide the final answer.\n"
                "- Resolve conflicts and ensure constraints are satisfied.\n"
                "- Use enum keys exactly.\n"
            )
        raise ValueError(f"unknown role: {role}")

    def task_block(self, stage: StageEnum, role: RoleEnum):
        if role == RoleEnum.PROPOSER:
            return self.task_textout_proposer(stage)
        if role == RoleEnum.REBUTTER:
            return self.task_textout_rebutter(stage)
        if role == RoleEnum.JUDGE:
            return self.task_judge(stage)
        raise ValueError(f"unknown role: {role}")

    def task_textout_proposer(self, stage: StageEnum):
        return "[Task]\n" + self.textout_task_proposer(stage)

    def task_textout_rebutter(self, stage: StageEnum):
        return "[Task]\n" + self.textout_task_rebutter(stage)

    def textout_task_proposer(self, stage: StageEnum):
        if stage == StageEnum.WORK_ENVIRONMENT:
            return (
                "PROPOSER:\n"
                "- Choose exactly 1 WorkEnvironment key.\n"
                "- Justify briefly with visible evidence.\n"
                "- If mapping/labels seem insufficient, add a concise suggestion.\n"
            )
        if stage == StageEnum.HAZARD:
            return (
                "PROPOSER:\n"
                "- Using the given state, list all applicable HazardFactor keys.\n"
                "- Justify briefly with visible evidence.\n"
                "- If mapping/labels seem insufficient, add a concise suggestion.\n"
            )
        if stage == StageEnum.COMPLIANCE:
            return (
                "PROPOSER:\n"
                "- Using the given state, decide the FINAL required PPE list (PPEItem keys).\n"
                "- Justify briefly with visible evidence.\n"
                "- If mapping/labels seem insufficient, add a concise suggestion.\n"
            )
        if stage == StageEnum.WEARING:
            return (
                "PROPOSER:\n"
                "- Using state.required_ppe, decide worn=true/false for each PPE item.\n"
                "- Justify briefly with visible evidence.\n"
            )
        if stage == StageEnum.IMPROPER_WEARING:
            return (
                "PROPOSER:\n"
                "- Using state.wearing, consider ONLY items where worn=true.\n"
                "- For those worn=true items, set improper_wearing.worn=true if worn incorrectly, else false.\n"
                "- Do NOT include items with worn=false in improper_wearing list.\n"
                "- Justify briefly with visible evidence.\n"
            )
        raise ValueError(f"unknown stage: {stage}")

    def textout_task_rebutter(self, stage: StageEnum):
        common = (
            "REBUTTER:\n"
            "- Start your text with 'AGREE' or 'DISAGREE'.\n"
            "- You must reference the proposer's claim if it exists in the state.\n"
            "- If DISAGREE, you must provide corrected enum keys and brief evidence.\n"
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
        if stage == StageEnum.WORK_ENVIRONMENT:
            return (
                "[Task]\n"
                "- Output WorkEnvironmentOut.\n"
                "- Choose exactly 1 WorkEnvironment key.\n"
                "- Provide a short reason.\n"
                "- Output proposals as a list. Use [] if no change is needed.\n"
            )
        if stage == StageEnum.HAZARD:
            return (
                "[Task]\n"
                "- Output HazardOut.\n"
                "- Select all applicable HazardFactor keys.\n"
                "- Provide a short reason.\n"
                "- Output proposals as a list. Use [] if no change is needed.\n"
            )
        if stage == StageEnum.COMPLIANCE:
            return (
                "[Task]\n"
                "- Output ComplianceOut.\n"
                "- Decide required_ppe as FINAL PPE list.\n"
                "- Provide a short reason.\n"
                "- Output proposals as a list. Use [] if no change is needed.\n"
            )
        if stage == StageEnum.WEARING:
            return (
                "[Task]\n"
                "- Output WearingOut.\n"
                "- Using state.required_ppe, decide worn=true/false for each PPE item.\n"
                "- Provide a short reason.\n"
            )
        if stage == StageEnum.IMPROPER_WEARING:
            return (
                "[Task]\n"
                "- Output ImproperWearingOut.\n"
                "- Consider ONLY items in state.wearing where worn=true.\n"
                "- Output improper_wearing list ONLY for those items (do not include worn=false items).\n"
                "- improper_wearing.worn=true means 'worn but incorrectly'; false means 'worn correctly'.\n"
                "- Provide a short reason.\n"
            )
        raise ValueError(f"unknown stage: {stage}")

    # -----------------
    # compact state
    # -----------------
    def compact_state_for_stage(self, stage: StageEnum, role: RoleEnum, round_idx: int, state: dict[str, Any]):
        base = {}

        # REBUTTER는 같은 round의 proposer_text를 state에 넣어줘야 함
        if role == RoleEnum.REBUTTER:
            if "proposer_text" in state:
                base["proposer_text"] = state["proposer_text"]


        if stage == StageEnum.WORK_ENVIRONMENT:
            return base  # (보통 {} / 또는 proposer_text만)

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
