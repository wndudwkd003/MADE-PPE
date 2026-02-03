# params/prompt_params.py

from enum import Enum


class RoleEnum(str, Enum):
    PROPOSER = "proposer"
    REBUTTER = "rebutter"
    JUDGE = "judge"


class StageEnum(str, Enum):
    WORK_ENVIRONMENT = "work_environment"
    HAZARD = "hazard"
    COMPLIANCE = "compliance"
    WEARING = "wearing"
    IMPROPER_WEARING = "improper_wearing"
