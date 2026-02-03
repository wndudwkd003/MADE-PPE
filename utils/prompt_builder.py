# utils/prompt_builder.py

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from params.prompt_params import RoleEnum, StageEnum
from params.structure import WorkEnvironment, HazardFactor, PPEItem
from params.ppe_mapping import WORK_ENV_TO_HAZARD_PPE


def enum_values(enum_cls) -> list[str]:
    return [e.value for e in enum_cls]


def mapping_text() -> str:
    lines: list[str] = []
    for we, hps in WORK_ENV_TO_HAZARD_PPE.items():
        lines.append(f"- {we.value}")
        for hp in hps:
            ppes = ", ".join([p.value for p in hp.ppe])
            lines.append(f"  - {hp.hazard.value}: {ppes}")
    return "\n".join(lines)


@dataclass(frozen=True)
class PromptPack:
    system: str
    user: str


class PromptBuilder:
    def __init__(self, include_mapping: bool = True):
        self.allowed_work_env = enum_values(WorkEnvironment)
        self.allowed_hazards = enum_values(HazardFactor)
        self.allowed_ppe = enum_values(PPEItem)

        self.include_mapping = include_mapping
        self.schema_text = mapping_text() if include_mapping else ""

    def build(
        self, stage: StageEnum, role: RoleEnum, state: dict[str, Any]
    ) -> PromptPack:
        system = self.system_prompt(role)
        user = self.user_prompt(stage, role, state)
        return PromptPack(system=system, user=user)

    def system_prompt(self, role: RoleEnum) -> str:
        base = (
            "You are an industrial safety labeling assistant for PPE compliance.\n"
            "Follow allowed labels strictly.\n"
            "Output must be a single JSON object that matches the given JSON Schema.\n"
            "Do not add extra keys. Do not wrap in markdown.\n"
        )

        if role == RoleEnum.PROPOSER:
            return (
                base
                + "Role: Proposer. Produce the best initial prediction with short evidence.\n"
            )
        if role == RoleEnum.REBUTTER:
            return (
                base
                + "Role: Rebutter. Critique the proposer and propose corrections.\n"
            )
        if role == RoleEnum.JUDGE:
            return base + "Role: Judge. Decide the final answer, resolving conflicts.\n"

        raise ValueError(f"unknown role: {role}")

    def user_prompt(
        self, stage: StageEnum, role: RoleEnum, state: dict[str, Any]
    ) -> str:
        state_json = json.dumps(state, ensure_ascii=False)

        rules = (
            "Allowed labels:\n"
            f"- WorkEnvironment: {self.allowed_work_env}\n"
            f"- HazardFactor: {self.allowed_hazards}\n"
            f"- PPEItem: {self.allowed_ppe}\n\n"
        )

        if self.include_mapping:
            rules += (
                "Reference mapping (WorkEnvironment -> HazardFactor -> Required PPE):\n"
                f"{self.schema_text}\n\n"
            )

        rules += f"State (previous outputs): {state_json}\n\n"

        return rules + self.task_text(stage, role)

    def task_text(self, stage: StageEnum, role: RoleEnum) -> str:
        if stage == StageEnum.WORK_ENVIRONMENT:
            return (
                "Task:\n"
                "- Choose exactly 1 WorkEnvironment label that best matches the image.\n"
                "- Provide a short reason.\n"
            )

        if stage == StageEnum.HAZARD:
            return (
                "Task:\n"
                "- Using state.work_environment, select all applicable HazardFactor labels.\n"
                "- Provide a short reason.\n"
            )

        if stage == StageEnum.COMPLIANCE:
            return (
                "Task:\n"
                "- Using state.work_environment and state.hazards, derive REQUIRED PPE based on the reference mapping.\n"
                "- Provide a short reason.\n"
            )

        if stage == StageEnum.WEARING:
            return (
                "Task:\n"
                "- Using state.required_ppe, decide for each PPE item whether it is worn (true/false).\n"
                "- Provide a short reason.\n"
            )

        if stage == StageEnum.IMPROPER_WEARING:
            return (
                "Task:\n"
                "- Using state.required_ppe and state.wearing, for each PPE item decide whether it is improperly worn.\n"
                "- Provide a short reason.\n"
            )

        raise ValueError(f"unknown stage: {stage}")
