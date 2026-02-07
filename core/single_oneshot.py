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
    """

    def __init__(self, config):
        super().__init__(config)
        from utils.prompt_builder_single_oneshot import SingleOneShotPromptBuilder
        self.prompt_builder = SingleOneShotPromptBuilder()

    def run_one_image(self, image_file_id, image_path, split, all_dir, labels_dir, errors_jsonl):
        cfg = self.config

        state = {}
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

    # -----------------
    # state update
    # -----------------
    def update_state_from_oneshot(self, parsed: OneShotOut, state: dict):
        # 1) WORK_ENVIRONMENT (WorkEnvironmentOut)
        we = parsed.work_environment
        state["work_environment"] = self._enum_to_name(we.work_environment)
        state["work_environment_reason"] = we.reason
        state["proposal_work_environment"] = we.proposal.model_dump() if we.proposal else None

        # 2) HAZARD (HazardOut)
        hz = parsed.hazard
        state["hazards"] = self._list_enum_to_names(hz.hazards)
        state["hazards_reason"] = hz.reason
        state["proposal_hazard"] = hz.proposal.model_dump() if hz.proposal else None

        # 3) COMPLIANCE (ComplianceOut)
        cp = parsed.compliance
        state["required_ppe"] = self._list_enum_to_names(cp.required_ppe)
        state["required_ppe_reason"] = cp.reason
        state["proposal_compliance"] = cp.proposal.model_dump() if cp.proposal else None

        # 4) WEARING (WearingOut)
        wg = parsed.wearing.model_dump()
        state["wearing"] = self._normalize_wearing_list(wg.get("wearing", []))
        state["wearing_reason"] = parsed.wearing.reason

        # 5) IMPROPER_WEARING (ImproperWearingOut)
        iw = parsed.improper_wearing.model_dump()
        state["improper_wearing"] = self._filter_improper_to_worn_only(
            iw.get("improper_wearing", []),
            state.get("wearing", []),
        )
        state["improper_wearing_reason"] = parsed.improper_wearing.reason
