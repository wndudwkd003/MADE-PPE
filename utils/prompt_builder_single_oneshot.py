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
    """

    def __init__(self):
        self.work_env_keys = enum_keys(WorkEnvironment)
        self.hazard_keys = enum_keys(HazardFactor)
        self.ppe_keys = enum_keys(PPEItem)

    # -----------------
    # public
    # -----------------
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
            "You are an industrial safety labeling assistant for Personal Protective Equipment (PPE) compliance. \n"
            "You will be given an image.\n"
            "Follow constraints and role instructions strictly.\n"
            "Keep reasons short and evidence-based.\n"
            "This is SINGLE-ONESHOT labeling: produce ALL stage outputs in ONE JSON.\n"
        )
    
    
    
