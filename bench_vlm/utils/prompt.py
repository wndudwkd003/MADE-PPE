# benchmark_vlm/core/prompts.py

from __future__ import annotations
import json
from typing import Any

SYSTEM_PROMPT = (
    "You are an industrial safety labeling assistant for Personal Protective Equipment (PPE) compliance.\n"
    "You will be given an image.\n"
    "Return ONLY a JSON object (no extra text).\n"
)

def make_user_prompt() -> str:
    schema = {
        "work_environment": "<WorkEnvironment>",
        "hazards": ["<HazardFactor>", "..."],
        "required_ppe": ["<PPEItem>", "..."],
        "wearing": [{"ppe": "<PPEItem>", "worn": True}],
        "improper_wearing": [{"ppe": "<PPEItem>", "worn": True}],
    }
    return (
        "[Task]\n"
        "Predict PPE compliance labels from the image.\n"
        "Output JSON must match this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
        "\n"
        "Rules:\n"
        "- hazards / required_ppe can be empty lists.\n"
        "- wearing and improper_wearing must be lists of {ppe, worn}.\n"
        "- improper_wearing should include ONLY PPE items that are worn=true in wearing.\n"
    )