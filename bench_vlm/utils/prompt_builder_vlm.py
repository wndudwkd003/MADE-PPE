# bench_vlm/utils/prompt_builder_vlm.py

from __future__ import annotations

import json
from collections import defaultdict

from params.params import TaskType
from params.structure import WorkEnvironment, HazardFactor, PPEItem
from params.ppe_mapping import WORK_ENV_TO_HAZARD_PPE


class VlmPromptBuilder:
    def __init__(self, task_mode: TaskType = TaskType.ALL_5STAGE):
        self.task_mode = task_mode

        self.envs = [e.name for e in WorkEnvironment]
        self.hazards = [h.name for h in HazardFactor]
        self.ppes = [p.name for p in PPEItem]

        self.env_relation = self._build_env_relation()

    def _build_env_relation(self) -> dict[str, dict[str, list[str]]]:
        out: dict[str, dict[str, list[str]]] = {}
        for env, hp_list in WORK_ENV_TO_HAZARD_PPE.items():
            hz_map: dict[str, list[str]] = {}
            for hp in hp_list:
                hz = hp.hazard.name
                hz_map[hz] = [p.name for p in hp.ppe]
            out[env.name] = hz_map
        return out

    def _compact_relation_text(self) -> str:
        parts = []
        for env in self.envs:
            hz_map = self.env_relation.get(env, {})
            inner = []
            for hz, ppes in hz_map.items():
                inner.append(f"{hz}:[{','.join(ppes)}]")
            parts.append(f"{env}:{{{','.join(inner)}}}")
        return "\n".join(parts)

    def build_prompt(self) -> str:
        base_rules = (
            "You are an industrial safety VLM.\n"
            "Output labels EXACTLY in the dataset schema using ONLY the provided code lists.\n\n"
            "STRICT RULES:\n"
            "1) Output ONLY a valid JSON object. No markdown. No ``` fences. No extra text.\n"
            "2) Use ONLY the codes from the lists below. Do NOT invent new strings.\n"
            "3) required_ppe MUST be consistent with the RELATION TABLE (WorkEnvironment→Hazard→PPE).\n"
            "4) wearing MUST be a LIST of objects: [{\"ppe\":\"PPE_CODE\",\"worn\":true/false}, ...]\n"
            "5) improper_wearing MUST be a LIST of objects (can be empty []): "
            "[{\"ppe\":\"PPE_CODE\",\"worn\":true/false}, ...]\n"
            "   - Include PPE that are worn but worn incorrectly.\n"
        )

        env_list = json.dumps(self.envs, ensure_ascii=False)
        hz_list = json.dumps(self.hazards, ensure_ascii=False)
        ppe_list = json.dumps(self.ppes, ensure_ascii=False)

        schema = (
            '{'
            '"work_environment":"WORK_ENV_CODE",'
            '"hazards":["HAZARD_CODE"],'
            '"required_ppe":["PPE_CODE"],'
            '"wearing":[{"ppe":"PPE_CODE","worn":true}],'
            '"improper_wearing":[{"ppe":"PPE_CODE","worn":true}]'
            '}'
        )

        if self.task_mode == TaskType.ALL_5STAGE:
            return (
                f"{base_rules}\n"
                f"ALLOWED work_environment codes:\n{env_list}\n\n"
                f"ALLOWED hazard codes:\n{hz_list}\n\n"
                f"ALLOWED PPE codes:\n{ppe_list}\n\n"
                "RELATION TABLE (ENV:{HAZARD:[PPE,...],...}):\n"
                f"{self._compact_relation_text()}\n\n"
                f"OUTPUT JSON SCHEMA:\n{schema}\n"
            )

        if self.task_mode == TaskType.SCENE:
            return (
                f"{base_rules}\n"
                f"Task: SCENE\nAllowed work_environment codes:\n{env_list}\n\n"
                "Output JSON schema:\n"
                '{"work_environment":"WORK_ENV_CODE"}\n'
            )

        if self.task_mode == TaskType.HAZARD:
            return (
                f"{base_rules}\n"
                f"Task: HAZARD\nAllowed hazard codes:\n{hz_list}\n\n"
                "Output JSON schema:\n"
                '{"hazards":["HAZARD_CODE"]}\n'
            )

        if self.task_mode == TaskType.REQUIRED:
            return (
                f"{base_rules}\n"
                f"Task: REQUIRED\nAllowed PPE codes:\n{ppe_list}\n\n"
                "Output JSON schema:\n"
                '{"required_ppe":["PPE_CODE"]}\n'
            )

        if self.task_mode == TaskType.WEARING:
            return (
                f"{base_rules}\n"
                f"Task: WEARING\nAllowed PPE codes:\n{ppe_list}\n\n"
                "Output JSON schema:\n"
                '{"wearing":[{"ppe":"PPE_CODE","worn":true}]}\n'
            )

        if self.task_mode == TaskType.IMPROPER:
            return (
                f"{base_rules}\n"
                f"Task: IMPROPER\nAllowed PPE codes:\n{ppe_list}\n\n"
                "Output JSON schema:\n"
                '{"improper_wearing":[]}\n'
            )

        return base_rules

    def build_target(self, label: dict) -> str:
        # IMPORTANT: match dataset labeling schema 그대로!
        if self.task_mode == TaskType.ALL_5STAGE:
            keys = ["work_environment", "hazards", "required_ppe", "wearing", "improper_wearing"]
            target_obj = {k: label.get(k) for k in keys}
            return json.dumps(target_obj, ensure_ascii=False)

        if self.task_mode == TaskType.SCENE:
            return json.dumps({"work_environment": label.get("work_environment")}, ensure_ascii=False)

        if self.task_mode == TaskType.HAZARD:
            return json.dumps({"hazards": label.get("hazards", [])}, ensure_ascii=False)

        if self.task_mode == TaskType.REQUIRED:
            return json.dumps({"required_ppe": label.get("required_ppe", [])}, ensure_ascii=False)

        if self.task_mode == TaskType.WEARING:
            return json.dumps({"wearing": label.get("wearing", [])}, ensure_ascii=False)

        if self.task_mode == TaskType.IMPROPER:
            return json.dumps({"improper_wearing": label.get("improper_wearing", [])}, ensure_ascii=False)

        return json.dumps({}, ensure_ascii=False)
