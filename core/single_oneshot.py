# core/single_oneshot.py

from __future__ import annotations

import time
from pathlib import Path
from tqdm.auto import tqdm

from core.agent import Agent
from params.output_schema import OneShotOut


class SingleOneShot(Agent):
    """
    SINGLE_ONESHOT:
      - ONE API call per image
      - outputs compatible with SINGLE_STEP/MADE label payload format
      - final_state includes stage_outputs for Analyzer compatibility
    """

    def __init__(self, config):
        super().__init__(config)
        from utils.prompt_builder_single_oneshot import SingleOneShotPromptBuilder
        self.prompt_builder = SingleOneShotPromptBuilder()

    def run_one_image(self, image_file_id, image_path, split, all_dir, labels_dir, errors_jsonl):
        cfg = self.config

        state = {"stage_outputs": {}}
        logs = []

        stem = image_path.stem
        all_final_path = Path(all_dir) / f"{stem}.json"
        labels_path = Path(labels_dir) / f"{stem}.json"

        # ---- single prompt ----
        pack = self.prompt_builder.build(state=state)

        tqdm.write("=" * 50)
        tqdm.write(f"[{stem}] SINGLE_ONESHOT START")
        tqdm.write("=" * 50)
        tqdm.write("-" * 50)
        tqdm.write(f"[{stem}] [ONESHOT] SYSTEM:\n{pack.system}\n")
        tqdm.write(f"[{stem}] [ONESHOT] USER:\n{pack.user}\n")
        tqdm.write("-" * 50)

        t0 = time.time()
        parsed_obj, last_err = self.api.call_responses(
            model=cfg.model.value,
            system_text=pack.system,
            user_text=pack.user,
            image_file_id=image_file_id,
            text_format=OneShotOut,
            max_output_tokens=cfg.max_output_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            retry_times=cfg.retry_times,
        )
        dt = time.time() - t0

        logs.append({
            "stage": "ONESHOT",
            "model": cfg.model.value,
            "text_format": getattr(OneShotOut, "__name__", str(OneShotOut)),
            "prompt": {"system": pack.system, "user": pack.user},
            "response": {
                "parsed": parsed_obj.model_dump() if parsed_obj is not None else None,
                "last_err": last_err,
            },
            "latency_sec": dt,
        })

        if parsed_obj is None:
            tqdm.write(f"[{stem}] [ONESHOT] PARSE_FAIL: {last_err}\n")
            self.append_error(errors_jsonl, {
                "ts": time.time(),
                "split": split,
                "image_path": str(image_path),
                "image_file_id": image_file_id,
                "where": "SingleOneShot.run_one_image",
                "stage": "ONESHOT",
                "error": last_err,
            })
            return None

        tqdm.write("-" * 50)
        tqdm.write(f"[{stem}] [ONESHOT] PARSED:\n{parsed_obj.model_dump()}\n")
        tqdm.write("-" * 50)

        # ---- update state ----
        self.update_state_from_oneshot(parsed_obj, state)

        # ---- save outputs ----
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

    def _dump_proposals(self, proposals):
        out = []
        for p in proposals:
            out.append(p.model_dump())
        return out

    # -----------------
    # state update
    # -----------------
    def update_state_from_oneshot(self, parsed: OneShotOut, state: dict):
        stage_outputs = state["stage_outputs"]

        # 1) WORK_ENVIRONMENT (WorkEnvironmentOut)
        we_out = parsed.work_environment
        we = self._enum_to_name(we_out.work_environment)
        we_reason = we_out.reason
        we_proposals = self._dump_proposals(we_out.proposals)

        state["work_environment"] = we
        state["work_environment_reason"] = we_reason
        state["proposals_work_environment"] = we_proposals

        stage_outputs["work_environment"] = {
            "work_environment": we,
            "reason": we_reason,
            "proposals": we_proposals,
        }

        # 2) HAZARD (HazardOut)
        hz_out = parsed.hazard
        hazards = self._list_enum_to_names(hz_out.hazards)
        hz_reason = hz_out.reason
        hz_proposals = self._dump_proposals(hz_out.proposals)

        state["hazards"] = hazards
        state["hazards_reason"] = hz_reason
        state["proposals_hazard"] = hz_proposals

        stage_outputs["hazard"] = {
            "hazards": hazards,
            "reason": hz_reason,
            "proposals": hz_proposals,
        }

        # 3) COMPLIANCE (ComplianceOut)
        cp_out = parsed.compliance
        required_ppe = self._list_enum_to_names(cp_out.required_ppe)
        cp_reason = cp_out.reason
        cp_proposals = self._dump_proposals(cp_out.proposals)

        state["required_ppe"] = required_ppe
        state["required_ppe_reason"] = cp_reason
        state["proposals_compliance"] = cp_proposals

        stage_outputs["compliance"] = {
            "required_ppe": required_ppe,
            "reason": cp_reason,
            "proposals": cp_proposals,
        }

        # 4) WEARING (WearingOut)
        wg = parsed.wearing.model_dump()
        wearing = self._normalize_wearing_list(wg.get("wearing", []))
        wg_reason = parsed.wearing.reason

        state["wearing"] = wearing
        state["wearing_reason"] = wg_reason

        stage_outputs["wearing"] = {
            "wearing": wearing,
            "reason": wg_reason,
        }

        # 5) IMPROPER_WEARING (ImproperWearingOut)
        iw = parsed.improper_wearing.model_dump()
        improper_raw = iw.get("improper_wearing", [])
        improper = self._filter_improper_to_worn_only(
            improper_raw,
            state.get("wearing", []),
        )
        iw_reason = parsed.improper_wearing.reason

        state["improper_wearing"] = improper
        state["improper_wearing_reason"] = iw_reason

        stage_outputs["improper_wearing"] = {
            "improper_wearing": improper,
            "reason": iw_reason,
        }
