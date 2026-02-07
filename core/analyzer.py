# core/analyzer.py

import json
import re
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path
from collections import Counter, defaultdict

from sentence_transformers import SentenceTransformer
from scipy.stats import chi2_contingency, fisher_exact

from config.config import Config
from utils.eval_utils import get_test_targets, get_all_samples, get_image_id
from typing import Any

@dataclass
class ProposalRecord:
    proposal_field: str
    proposal_type: str # add, remove, modify ...
    target_from: str
    target_to: str
    proposal_text: str


class Analyzer:
    def __init__(self, config: Config):
        self.config = config

        self.embed_model = SentenceTransformer(
            config.sentence_emb_model,
            device=config.device,
        )


    def run(self):
        started_at = datetime.now().isoformat(timespec="seconds")

        target_paths = get_test_targets(
            agent=self.config.agent,
            dataset=self.config.dataset,
            targets=self.config.test_targets,
            run_dir=self.config.runs,
        )

        all_records = []
        num_samples_read = 0

        sample_lookup = {}

        for path in target_paths:
            target_tag = path.name
            samples_by_split = get_all_samples(path)

            for split, samples in samples_by_split.items():
                num_samples_read += len(samples)

                for sample in samples:
                    image_path = sample["image"]
                    image_id = get_image_id(image_path)

                    key = (target_tag, split, image_id)
                    sample_lookup[key] = sample

                    recs = self.extract_records(sample)

                    if len(recs) > 0:
                        all_records.append({
                            "target_tag": target_tag,
                            "split": split,
                            "image_id": image_id,
                            "proposed_records": recs,
                        })







    def extract_records(self, sample: dict):
        final_state = sample["final_state"]
        proposals = self.get_proposal(final_state)

        records = []
        for pk, pv in proposals:
            proposal_field = pk[len("proposal_") :]
            pv_type = pv["type"]
            proposal_text = pv["proposal"]
            target_to = pv["target_to"]

            if pv_type == "add":
                target_from = ""

            elif pv_type == "modify":
                target_from = pv.get("target_from") or final_state.get(proposal_field, "")

            elif pv_type == "remove":
                target_from = pv.get("target_from") or final_state.get(proposal_field, "")

            else:
                target_from = pv.get("target_from", "")

            records.append(
                ProposalRecord(
                    proposal_field=proposal_field,
                    proposal_type=pv_type,
                    target_from=target_from,
                    target_to=target_to,
                    proposal_text=proposal_text,
                )
            )
        return records


    def get_proposal(self, final_state: dict[str, Any]):
        proposals = []
        for key, value in final_state.items():
            if not key.startswith("proposal_"):
                continue

            if value["flag"] is not True:
                continue

            proposals.append((key, value))

        return proposals
