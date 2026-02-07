# core/single_step.py

from __future__ import annotations

from pathlib import Path
import time
from tqdm.auto import tqdm

from core.agent import Agent
from params.prompt_params import StageEnum
from params.output_schema import (
    WorkEnvironmentOut,
    HazardOut,
    ComplianceOut,
    WearingOut,
    ImproperWearingOut,
)


class SingleStep(Agent):
    """
    Stage별 single agent 1회 호출로 최종 스키마 출력 생성.
    - output schema는 MADE의 Judge 출력과 동일하게 유지하여 공정 비교가 가능하도록 함.
    """

    def __init__(self, config):
        super().__init__(config)
        from utils.prompt_builder_single_step import SingleStepPromptBuilder
        self.prompt_builder = SingleStepPromptBuilder()

    def run_one_image(self, image_file_id, image_path, split, all_dir, labels_dir, errors_jsonl):
        cfg = self.config

        stages = [
            StageEnum.WORK_ENVIRONMENT,
            StageEnum.HAZARD,
            StageEnum.COMPLIANCE,
            StageEnum.WEARING,
            StageEnum.IMPROPER_WEARING,
        ]

        state = {}
        logs = []

        stem = image_path.stem
        all_final_path = all_dir / f"{stem}.json"
        labels_path = labels_dir / f"{stem}.json"

        for stage in stages:
            tqdm.write("=" * 50)
            tqdm.write(f"[{image_path.stem}] SINGLE_STEP START STAGE: {stage.name}")
            tqdm.write("=" * 50)

            pack = self.prompt_builder.build(stage=stage, state=state)

            tqdm.write("-" * 50)
            tqdm.write(f"[{image_path.stem}] [{stage.name}] SYSTEM:\n{pack.system}\n")
            tqdm.write(f"[{image_path.stem}] [{stage.name}] USER:\n{pack.user}\n")
            tqdm.write("-" * 50)

            text_format = self.get_text_format(stage)

            t0 = time.time()
            parsed_obj, last_err = self.api.call_responses(
                model=cfg.model.value,
                system_text=pack.system,
                user_text=pack.user,
                image_file_id=image_file_id,
                text_format=text_format,
                max_output_tokens=cfg.max_output_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                retry_times=cfg.retry_times,
            )
            dt = time.time() - t0

            logs.append({
                "stage": stage.name,
                "model": cfg.model.value,
                "text_format": getattr(text_format, "__name__", str(text_format)),
                "prompt": {"system": pack.system, "user": pack.user},
                "response": {
                    "parsed": parsed_obj.model_dump() if parsed_obj is not None else None,
                    "last_err": last_err,
                },
                "latency_sec": dt,
            })

            if parsed_obj is None:
                tqdm.write(f"[{image_path.stem}] [{stage.name}] PARSE_FAIL: {last_err}\n")
                self.append_error(errors_jsonl, {
                    "ts": time.time(),
                    "split": split,
                    "image_path": str(image_path),
                    "image_file_id": image_file_id,
                    "where": "SingleStep.run_one_image",
                    "stage": stage.name,
                    "error": last_err,
                })
                return None

            tqdm.write("-" * 50)
            tqdm.write(f"[{image_path.stem}] [{stage.name}] PARSED:\n{parsed_obj.model_dump()}\n")
            tqdm.write("-" * 50)

            self.update_state(stage, parsed_obj, state)

        # ----- 여기까지 오면 "이미지 1장 완료" -----
        labels_payload = self._make_labels_payload(image_path, split, state)
        self.write_json(labels_path, labels_payload)

        all_payload = {
            "image": str(image_path),
            "split": split,
            "status": "done",
            "file_id": image_file_id,
            "final_state": state,
            "labels": labels_payload,
            "logs": logs,
        }
        self.write_json(all_final_path, all_payload)
        return all_payload
    
    # -----------------
    # outputs (labels)
    # -----------------
    def _make_labels_payload(self, image_path, split, state):
        return {
            "image": str(image_path),
            "split": split,
            "work_environment": state.get("work_environment"),
            "hazards": state.get("hazards", []),
            "required_ppe": state.get("required_ppe", []),
            "wearing": state.get("wearing", []),
            "improper_wearing": state.get("improper_wearing", []),
        }
    
    # -----------------
    # schema selection (pydantic)
    # -----------------
    def get_text_format(self, stage: StageEnum):
        if stage == StageEnum.WORK_ENVIRONMENT:
            return WorkEnvironmentOut
        if stage == StageEnum.HAZARD:
            return HazardOut
        if stage == StageEnum.COMPLIANCE:
            return ComplianceOut
        if stage == StageEnum.WEARING:
            return WearingOut
        if stage == StageEnum.IMPROPER_WEARING:
            return ImproperWearingOut
        raise ValueError(f"Unknown stage: {stage}")
    
    # -----------------
    # state update (same as MADE judge outputs)
    # -----------------
    def _enum_to_name(self, v):
        return v.name if hasattr(v, "name") else v

    def _list_enum_to_names(self, xs):
        return [self._enum_to_name(x) for x in xs]

    def _normalize_wearing_list(self, items):
        out = []
        for it in items:
            out.append({
                "ppe": self._enum_to_name(it.get("ppe")),
                "worn": it.get("worn"),
            })
        return out
    
    def update_state(self, stage: StageEnum, parsed, state: dict):
        # WORK_ENVIRONMENT
        if stage == StageEnum.WORK_ENVIRONMENT:
            state["work_environment"] = self._enum_to_name(parsed.work_environment)
            state["work_environment_reason"] = parsed.reason
            state["proposal_work_environment"] = parsed.proposal.model_dump() if parsed.proposal else None
            return

        # HAZARD
        if stage == StageEnum.HAZARD:
            state["hazards"] = self._list_enum_to_names(parsed.hazards)
            state["hazards_reason"] = parsed.reason
            state["proposal_hazard"] = parsed.proposal.model_dump() if parsed.proposal else None
            return

        # COMPLIANCE
        if stage == StageEnum.COMPLIANCE:
            state["required_ppe"] = self._list_enum_to_names(parsed.required_ppe)
            state["required_ppe_reason"] = parsed.reason
            state["proposal_compliance"] = parsed.proposal.model_dump() if parsed.proposal else None
            return

        # WEARING
        if stage == StageEnum.WEARING:
            d = parsed.model_dump()
            state["wearing"] = self._normalize_wearing_list(d["wearing"])
            state["wearing_reason"] = parsed.reason
            return

        # IMPROPER_WEARING
        if stage == StageEnum.IMPROPER_WEARING:
            d = parsed.model_dump()
            state["improper_wearing"] = self._filter_improper_to_worn_only(
                d.get("improper_wearing", []),
                state.get("wearing", []),
            )
            state["improper_wearing_reason"] = parsed.reason
            return

        raise ValueError(f"Unknown stage in update_state: {stage}")
    
    def _filter_improper_to_worn_only(self, improper_items, wearing_items):
        worn_set = set()
        for it in (wearing_items or []):
            ppe = it.get("ppe")
            if ppe and bool(it.get("worn")):
                worn_set.add(ppe)

        out = []
        for it in (improper_items or []):
            ppe = self._enum_to_name(it.get("ppe"))
            if not ppe:
                continue
            if ppe in worn_set:
                out.append({"ppe": ppe, "worn": bool(it.get("worn"))})
        return out