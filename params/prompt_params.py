# params/prompt_params.py

from dataclasses import dataclass

from enum import Enum


class RoleEnum(str, Enum):
    PROPOSER = "proposer"
    REBUTTER = "rebutter"
    JUDGE = "judge"


class StageEnum(str, Enum):
    WORK_ENVIRONMENT = "work_environment"
    HAZARD = "hazards"
    COMPLIANCE = "required_ppe"
    WEARING = "wearing"
    IMPROPER_WEARING = "improper_wearing"


@dataclass(frozen=True)
class PromptPack:
    system: str
    user: str
