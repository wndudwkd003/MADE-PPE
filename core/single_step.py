# core/single_step.py

from __future__ import annotations

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

        state = {"stage_outputs": {}}
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
    # helpers
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

    def _dump_proposals(self, proposals):
        # proposals는 ProposalOut의 리스트(Pydantic)이며, JSON 저장을 위해 dict로 변환해야 합니다.
        out = []
        for p in proposals:
            out.append(p.model_dump())
        return out

    # -----------------
    # state update (MADE judge outputs와 동일한 stage_outputs 구조 생성)
    # -----------------
    def update_state(self, stage: StageEnum, parsed, state: dict):
        stage_outputs = state["stage_outputs"]

        if stage == StageEnum.WORK_ENVIRONMENT:
            we = self._enum_to_name(parsed.work_environment)
            reason = parsed.reason
            proposals = self._dump_proposals(parsed.proposals)

            state["work_environment"] = we
            state["work_environment_reason"] = reason
            state["proposals_work_environment"] = proposals

            stage_outputs["work_environment"] = {
                "work_environment": we,
                "reason": reason,
                "proposals": proposals,
            }
            return

        if stage == StageEnum.HAZARD:
            hazards = self._list_enum_to_names(parsed.hazards)
            reason = parsed.reason
            proposals = self._dump_proposals(parsed.proposals)

            state["hazards"] = hazards
            state["hazards_reason"] = reason
            state["proposals_hazard"] = proposals

            stage_outputs["hazard"] = {
                "hazards": hazards,
                "reason": reason,
                "proposals": proposals,
            }
            return

        if stage == StageEnum.COMPLIANCE:
            required_ppe = self._list_enum_to_names(parsed.required_ppe)
            reason = parsed.reason
            proposals = self._dump_proposals(parsed.proposals)

            state["required_ppe"] = required_ppe
            state["required_ppe_reason"] = reason
            state["proposals_compliance"] = proposals

            stage_outputs["compliance"] = {
                "required_ppe": required_ppe,
                "reason": reason,
                "proposals": proposals,
            }
            return

        if stage == StageEnum.WEARING:
            d = parsed.model_dump()
            wearing = self._normalize_wearing_list(d["wearing"])
            reason = parsed.reason

            state["wearing"] = wearing
            state["wearing_reason"] = reason

            stage_outputs["wearing"] = {
                "wearing": wearing,
                "reason": reason,
            }
            return

        if stage == StageEnum.IMPROPER_WEARING:
            d = parsed.model_dump()
            improper = self._filter_improper_to_worn_only(
                d.get("improper_wearing", []),
                state.get("wearing", []),
            )
            reason = parsed.reason

            state["improper_wearing"] = improper
            state["improper_wearing_reason"] = reason

            stage_outputs["improper_wearing"] = {
                "improper_wearing": improper,
                "reason": reason,
            }
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
