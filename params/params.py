# params/params.py

from enum import Enum


class DatasetEnum(Enum):
    SH17 = "SH17"
    SCP300 = "SCP300"


class AgentEnum(Enum):
    MADE = "MADE"
    SINGLE_STEP = "SINGLE_STEP"
    SINGLE_ONESHOT = "SINGLE_ONESHOT"


class ModelEnum(Enum):
    GPT5_MINI = "gpt-5-mini-2025-08-07"
    GPT5_1 = "gpt-5.1-2025-11-13"
    GPT4_1 = "gpt-4.1-2025-04-14"
    CLAUDE_SONNET4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_HAIKU4_5 = "claude-haiku-4-5-20251001"


class DoModeEnum(Enum):
    LABELING = "labeling"
    EVALUATION = "evaluation"
    ANALYSIS = "analysis"


class TestModeEnum(Enum):
    MADE_PPE = "made_ppe"
    MADE_BENCH = "made_bench"
