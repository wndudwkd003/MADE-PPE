# core/made.py

from __future__ import annotations

from pathlib import Path
import time
from regex import B
from tqdm.auto import tqdm

from core.agent import Agent
from params.prompt_params import RoleEnum, StageEnum
from params.output_schema import (
    TextOut,
    WorkEnvironmentOut,
    HazardOut,
    ComplianceOut,
    WearingOut,
    ImproperWearingOut,
)


class MADE(Agent):
    def __init__(self, config):
        super().__init__(config)
        from utils.prompt_builder import PromptBuilder
        self.prompt_builder = PromptBuilder()

    def run_one_image(self, image_file_id, image_path, split, all_dir, labels_dir, errors_jsonl):
        cfg = self.config

        stages = [
            StageEnum.WORK_ENVIRONMENT,
            StageEnum.HAZARD,
            StageEnum.COMPLIANCE,
            StageEnum.WEARING,
            StageEnum.IMPROPER_WEARING,
        ]
        role_seq = cfg.role_sequence

        state = {"stage_outputs": {}}
        logs = []

        stem = image_path.stem
        all_final_path = all_dir / f"{stem}.json"
        labels_path = labels_dir / f"{stem}.json"

        for stage in stages:
            state["stage_history"] = []

            tqdm.write("=" * 50)
            tqdm.write(f"[{image_path.stem}] START STAGE: {stage.name}")
            tqdm.write("=" * 50)

            for role, round_idx in role_seq:
                text_format = self.get_text_format(stage, role)

                pack = self.prompt_builder.build(
                    stage=stage,
                    role=role,
                    round_idx=round_idx,
                    state=state,
                    text_format=text_format,
                )

                tqdm.write("-" * 50)
                tqdm.write(f"[{image_path.stem}] [{stage.name}] [{role.name} r{round_idx}] SYSTEM:\n{pack.system}\n")
                tqdm.write(f"[{image_path.stem}] [{stage.name}] [{role.name} r{round_idx}] USER:\n{pack.user}\n")
                tqdm.write("-" * 50)



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
                    "role": role.name,
                    "round": round_idx,
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
                    tqdm.write(f"[{image_path.stem}] [{stage.name}] [{role.name} r{round_idx}] PARSE_FAIL: {last_err}\n")
                    self.append_error(errors_jsonl, {
                        "ts": time.time(),
                        "split": split,
                        "image_path": str(image_path),
                        "image_file_id": image_file_id,
                        "where": "MADE.run_one_image",
                        "stage": stage.name,
                        "role": role.name,
                        "round": round_idx,
                        "error": last_err,
                    })
                    return None

                tqdm.write("-" * 50)
                tqdm.write(f"[{image_path.stem}] [{stage.name}] [{role.name} r{round_idx}] PARSED:\n{parsed_obj.model_dump()}\n")
                tqdm.write("-" * 50)
                tqdm.write("-" * 50)

                # state 업데이트 (proposer_text 단일 키 기준)
                self.update_state(stage, role, round_idx, parsed_obj, state)

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

    def get_text_format(self, stage, role):
        if role in (RoleEnum.PROPOSER, RoleEnum.REBUTTER):
            return TextOut
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
        raise ValueError(f"Unknown stage/role: {stage}, {role}")

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

    def update_state(
        self,
        stage: StageEnum,
        role: RoleEnum,
        round_idx: int,
        parsed,
        state: dict
    ):
        if role in (RoleEnum.PROPOSER, RoleEnum.REBUTTER):
            state["stage_history"].append({
                "round": round_idx,
                "role": role.name,
                "text": parsed.text
            })
            return

        stage_key = stage.name.lower()
        state[f"{stage_key}_reason"] = parsed.reason
        d = parsed.model_dump()

        # 제안(Proposals) 저장
        if hasattr(parsed, "proposals"):
            state[f"proposals_{stage_key}"] = [p.model_dump() for p in parsed.proposals]

        # 스테이지별 전용 데이터 추출 로직
        if stage == StageEnum.WORK_ENVIRONMENT:
            state["work_environment"] = self._enum_to_name(parsed.work_environment)

        elif stage == StageEnum.HAZARD:
            state["hazards"] = self._list_enum_to_names(parsed.hazards)

        elif stage == StageEnum.COMPLIANCE:
            state["required_ppe"] = self._list_enum_to_names(parsed.required_ppe)

        elif stage == StageEnum.WEARING:
            state["wearing"] = self._normalize_wearing_list(d.get("wearing", []))

        elif stage == StageEnum.IMPROPER_WEARING:
            # 착용(worn=true)된 것들만 필터링하는 기존 로직 유지
            raw_improper = d.get("improper_wearing", [])
            state["improper_wearing"] = self._filter_improper_to_worn_only(
                raw_improper,
                state.get("wearing", [])
            )

        state["stage_outputs"][stage_key] = d

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
