# core/analyzer.py

from __future__ import annotations

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
from utils.analysis_utils import get_proposal


SCIPY_AVAILABLE = True


class DisjointSet:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class ProposalRecord:
    target_tag: str
    split: str
    image_id: str
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

        self.stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
            "is", "are", "was", "were", "be", "been", "being",
            "this", "that", "these", "those",
            "it", "its", "they", "them", "their", "there",
            "of", "to", "in", "on", "for", "with", "as", "at", "by", "from", "into",
            "not", "no", "nor",
            "image", "depicts", "shows", "scene", "environment", "category", "labels", "covered", "current",
        }

        # proposal_field -> final_state key 매핑(필요 최소만)
        self.field_to_final_key = {
            "work_environment": "work_environment",
            "hazards": "hazards",
            "hazard": "hazards",
            "required_ppe": "required_ppe",
            "compliance": "compliance",
        }

    # ---------- small helpers ----------
    def densify_grouped(self, grouped: dict, type_keys: list[str]):
        out = {}
        for group_key, score_map in grouped.items():
            fixed = dict(score_map)
            for t in type_keys:
                if t not in fixed:
                    fixed[t] = 0
            out[group_key] = fixed
        return out

    def transition_to_text(self, s: str):
        # "A -> B" 를 자연어로 바꿔 embedding 품질을 올림
        if "->" in s:
            left, right = s.split("->", 1)
            left = left.strip().replace("_", " ").lower()
            right = right.strip().replace("_", " ").lower()
            return f"{left} to {right}"
        return s.replace("_", " ").lower()

    def cosine_similarity(self, a, b):
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(len(a)):
            va = float(a[i])
            vb = float(b[i])
            dot += va * vb
            na += va * va
            nb += vb * vb
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))

    def pick_more_specific(self, items: list[str]):
        # “명확한쪽” = 더 구체적/정보가 많은쪽을 대표로 선택
        best = items[0]
        best_score = -1.0
        for s in items:
            tokens = [t for t in s.split("_") if t]
            score = float(len(tokens)) * 10.0 + float(len(s))
            if score > best_score:
                best_score = score
                best = s
        return best

    def merge_similar_text_counts(self, count_map: dict[str, int], threshold: float):
        keys = list(count_map.keys())
        if len(keys) == 0:
            return {}, {"clusters": [], "threshold": float(threshold)}

        texts = [self.transition_to_text(k) for k in keys]
        embs = self.embed_model.encode(texts, normalize_embeddings=False)

        dsu = DisjointSet(len(keys))

        for i in range(len(keys)):
            ei = embs[i]
            for j in range(i + 1, len(keys)):
                sim = self.cosine_similarity(ei, embs[j])
                if sim >= threshold:
                    dsu.union(i, j)

        clusters = {}
        for i in range(len(keys)):
            r = dsu.find(i)
            if r in clusters:
                clusters[r].append(i)
            else:
                clusters[r] = [i]

        merged = {}
        debug_clusters = []

        for _, idxs in clusters.items():
            members = [keys[i] for i in idxs]
            rep = self.pick_more_specific(members)

            total = 0
            for m in members:
                total += int(count_map[m])

            merged[rep] = int(total)
            debug_clusters.append(
                {
                    "representative": rep,
                    "members": members,
                    "total_count": int(total),
                }
            )

        merged_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        merged = {k: int(v) for k, v in merged_items}

        return merged, {"clusters": debug_clusters, "threshold": float(threshold)}

    # ---------- export ----------
    def export_flagged_samples(self, grouped_samples: dict, sample_lookup: dict):
        """
        proposal이 있는 샘플만 eval_runs 아래로 이미지+json 재저장.
        - grouped_samples: {(target_tag, split, image_id): [ProposalRecord, ...]}
        - sample_lookup:  {(target_tag, split, image_id): sample_dict}
        """
        root = Path(self.config.eval_runs) / "proposal_exports" / str(self.config.dataset) / str(self.config.agent)

        jsonl_by_split = defaultdict(list)

        for key, recs in grouped_samples.items():
            target_tag, split, image_id = key

            if len(recs) == 0:
                continue

            sample = sample_lookup[key]
            image_src = Path(sample["image"])

            dst_dir = root / str(target_tag) / str(split) / str(image_id)
            dst_dir.mkdir(parents=True, exist_ok=True)

            # image copy
            dst_image = dst_dir / image_src.name
            shutil.copy2(str(image_src), str(dst_image))

            # json per sample
            payload = {
                "target_tag": target_tag,
                "split": split,
                "image_id": image_id,
                "image_src": str(image_src),
                "image_dst": str(dst_image),
                "num_proposals": int(len(recs)),
                "proposals": [
                    {
                        "proposal_field": r.proposal_field,
                        "proposal_type": r.proposal_type,
                        "target_from": r.target_from,
                        "target_to": r.target_to,
                        "proposal_text": r.proposal_text,
                    }
                    for r in recs
                ],
            }

            with open(dst_dir / "sample.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            jsonl_by_split[(target_tag, split)].append(payload)

        # split별 jsonl 저장(한 파일로 보기 편하게)
        for (target_tag, split), rows in jsonl_by_split.items():
            out_dir = root / str(target_tag) / str(split)
            out_dir.mkdir(parents=True, exist_ok=True)

            with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False))
                    f.write("\n")

    # ---------- public ----------
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

        # export를 위해 “proposal 있는 샘플”을 다시 찾을 수 있게 lookup 구성
        sample_lookup = {}

        for path in target_paths:
            target_tag = path.name
            samples_by_split = get_all_samples(path)

            for split, samples in samples_by_split.items():
                num_samples_read += len(samples)

                for sample in samples:
                    image_id = get_image_id(sample["image"])
                    key = (target_tag, split, image_id)
                    sample_lookup[key] = sample





                    recs = self.extract_records(target_tag, split, sample)
                    if len(recs) > 0:
                        all_records.extend(recs)





        plot_counts = self.build_counts(all_records, top_k=self.config.top_k)
        type_keys = list(plot_counts["proposal_type"].keys())

        merged_transitions, merged_transitions_debug = self.merge_similar_text_counts(
            plot_counts["top_transitions"],
            threshold=0.86,
        )

        grouped_target_type = self.densify_grouped(
            self.grouped_by_target_type(all_records),
            type_keys,
        )
        grouped_field_type = self.densify_grouped(
            self.grouped_by_field_type(all_records),
            type_keys,
        )

        terms_all = self.top_terms(all_records, top_k=30)
        examples_all = self.sample_examples(all_records, n=30, seed=self.config.seed)

        by_type = defaultdict(list)
        for r in all_records:
            by_type[r.proposal_type].append(r)

        type_summaries = {}
        for t, recs in by_type.items():
            type_summaries[t] = {
                "count": int(len(recs)),
                "top_terms": self.top_terms(recs, top_k=15),
                "examples": self.sample_examples(recs, n=10, seed=self.config.seed + 7),
            }

        infer = self.analyze_inferential(all_records)

        grouped_samples = self.group_records_by_sample(all_records)
        sample_units = list(grouped_samples.values())

        ci_add = self.bootstrap_ci_for_share(sample_units, "add_share", iters=2000, seed=self.config.seed)
        ci_any = self.bootstrap_ci_for_share(sample_units, "has_any_proposal_rate", iters=2000, seed=self.config.seed + 11)

        per_target_samples = defaultdict(list)
        for (target_tag, _split, _image_id), recs in grouped_samples.items():
            per_target_samples[target_tag].append(recs)

        ci_by_target = {}
        for target_tag, units in per_target_samples.items():
            ci_by_target[target_tag] = {
                "add_share": self.bootstrap_ci_for_share(units, "add_share", iters=2000, seed=self.config.seed + 101),
                "has_any_proposal_rate": self.bootstrap_ci_for_share(units, "has_any_proposal_rate", iters=2000, seed=self.config.seed + 202),
            }

        # proposal 있는 샘플 export
        self.export_flagged_samples(grouped_samples, sample_lookup)

        summary = {
            "counts": {
                "num_samples_read": int(num_samples_read),
                "num_flagged_proposal_records": int(len(all_records)),
                "num_sample_units": int(len(sample_units)),
            },
            "descriptive": {
                "proposal_type_counts": plot_counts["proposal_type"],
                "proposal_field_counts": plot_counts["proposal_field"],
                "top_transitions": merged_transitions,
                "top_from": plot_counts["top_from"],
                "top_to": plot_counts["top_to"],
            },
            "merge": {
                "top_transitions_semantic": merged_transitions_debug,
            },
            "grouped": {
                "by_target_type": grouped_target_type,
                "by_field_type": grouped_field_type,
                "type_keys": type_keys,
            },
            "text": {
                "top_terms_all": terms_all,
                "examples_all": examples_all,
                "by_type": type_summaries,
            },
            "inferential": infer,
            "bootstrap_ci": {
                "overall_add_share": ci_add,
                "overall_has_any_proposal_rate": ci_any,
                "by_target": ci_by_target,
            },
            "meta": {
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "scipy_available": SCIPY_AVAILABLE,
                "export_dir": str(Path(self.config.eval_runs) / "proposal_exports" / str(self.config.dataset) / str(self.config.agent)),
            },
        }

        return summary

    # ---------- record extraction ----------
    def extract_records(self, target_tag: str, split: str, sample: dict):
        final_state = sample["final_state"]
        proposals = get_proposal(final_state)

        records = []
        for pk, pv in proposals:
            proposal_field = pk[len("proposal_") :]

            target_from = pv["target_from"]
            if target_from == "":
                base_val = None

                if proposal_field in final_state:
                    base_val = final_state[proposal_field]
                elif proposal_field in self.field_to_final_key:
                    base_key = self.field_to_final_key[proposal_field]
                    if base_key in final_state:
                        base_val = final_state[base_key]

                if base_val is not None:
                    if isinstance(base_val, list):
                        target_from = "|".join([str(x) for x in base_val])
                    else:
                        target_from = str(base_val)
                else:
                    target_from = ""

            records.append(
                ProposalRecord(
                    target_tag=target_tag,
                    split=split,
                    image_id=get_image_id(sample["image"]),
                    proposal_field=proposal_field,
                    proposal_type=pv["type"],
                    target_from=target_from,
                    target_to=pv["target_to"],
                    proposal_text=pv["proposal"],
                )
            )

        return records

    # ---------- counters ----------
    def build_counts(self, records: list[ProposalRecord], top_k: int):
        type_counter = Counter(r.proposal_type for r in records if r.proposal_type)
        field_counter = Counter(r.proposal_field for r in records if r.proposal_field)
        from_counter = Counter(r.target_from for r in records if r.target_from)
        to_counter = Counter(r.target_to for r in records if r.target_to)
        trans_counter = Counter(
            f"{r.target_from} -> {r.target_to}"
            for r in records
            if r.target_from or r.target_to
        )

        return {
            "proposal_type": dict(type_counter.most_common()),
            "proposal_field": dict(field_counter.most_common()),
            "top_transitions": dict(trans_counter.most_common(top_k)),
            "top_from": dict(from_counter.most_common(top_k)),
            "top_to": dict(to_counter.most_common(top_k)),
        }

    def grouped_by_target_type(self, records: list[ProposalRecord]):
        per_target = defaultdict(Counter)
        for r in records:
            if r.target_tag and r.proposal_type:
                per_target[r.target_tag][r.proposal_type] += 1
        return {k: dict(v) for k, v in per_target.items()}

    def grouped_by_field_type(self, records: list[ProposalRecord]):
        per_field = defaultdict(Counter)
        for r in records:
            if r.proposal_field and r.proposal_type:
                per_field[r.proposal_field][r.proposal_type] += 1
        return {k: dict(v) for k, v in per_field.items()}

    # ---------- text ----------
    def tokenize(self, text: str):
        t = text.lower()
        t = re.sub(r"[^a-z0-9]+", " ", t)
        parts = t.split()

        out = []
        for p in parts:
            if p in self.stopwords:
                continue
            if len(p) <= 2:
                continue
            out.append(p)
        return out

    def top_terms(self, records: list[ProposalRecord], top_k: int):
        counter = Counter()
        for r in records:
            if r.proposal_text:
                counter.update(self.tokenize(r.proposal_text))
        items = counter.most_common(top_k)
        return [{"term": term, "count": int(cnt)} for term, cnt in items]

    def sample_examples(self, records: list[ProposalRecord], n: int, seed: int):
        rnd = random.Random(seed)
        if len(records) == 0:
            return []

        idxs = list(range(len(records)))
        rnd.shuffle(idxs)
        idxs = idxs[:n]

        out = []
        for i in idxs:
            r = records[i]
            out.append(
                {
                    "target_tag": r.target_tag,
                    "split": r.split,
                    "image_id": r.image_id,
                    "proposal_field": r.proposal_field,
                    "proposal_type": r.proposal_type,
                    "target_from": r.target_from,
                    "target_to": r.target_to,
                    "proposal_text": r.proposal_text,
                }
            )
        return out

    # ---------- inferential ----------
    def build_contingency_table(self, records: list[ProposalRecord], row_attr: str, col_attr: str):
        row_vals = []
        col_vals = []

        for r in records:
            if row_attr == "target_tag":
                rv = r.target_tag
            elif row_attr == "split":
                rv = r.split
            elif row_attr == "proposal_field":
                rv = r.proposal_field
            else:
                rv = r.proposal_type

            if col_attr == "proposal_field":
                cv = r.proposal_field
            else:
                cv = r.proposal_type

            if rv and cv:
                row_vals.append(rv)
                col_vals.append(cv)

        row_keys = sorted(set(row_vals))
        col_keys = sorted(set(col_vals))

        row_index = {k: i for i, k in enumerate(row_keys)}
        col_index = {k: j for j, k in enumerate(col_keys)}

        table = [[0 for _ in col_keys] for _ in row_keys]
        for rv, cv in zip(row_vals, col_vals):
            table[row_index[rv]][col_index[cv]] += 1

        return row_keys, col_keys, table

    def cramers_v_from_table(self, table: list, chi2_stat: float):
        n = 0
        for row in table:
            for v in row:
                n += int(v)

        if n == 0:
            return 0.0

        r = len(table)
        c = len(table[0]) if r > 0 else 0
        if r <= 1 or c <= 1:
            return 0.0

        denom = float(n) * float(min(r - 1, c - 1))
        return sqrt(float(chi2_stat) / denom) if denom > 0.0 else 0.0

    def chisquare_test(self, table: list):
        chi2_stat, p_value, dof, _expected = chi2_contingency(table)
        return {"chi2": float(chi2_stat), "dof": int(dof), "p_value": float(p_value)}

    def fisher_test_2x2(self, a: int, b: int, c: int, d: int):
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
        return {"odds_ratio": float(odds_ratio), "p_value": float(p_value)}

    def bh_fdr(self, p_values: list[float]):
        indexed = list(enumerate(p_values))
        indexed.sort(key=lambda x: x[1])

        m = len(indexed)
        q_values = [0.0 for _ in range(m)]

        prev = 1.0
        for rank in range(m, 0, -1):
            idx, p = indexed[rank - 1]
            q = float(p) * float(m) / float(rank)
            if q > prev:
                q = prev
            prev = q
            q_values[idx] = q

        return q_values

    def analyze_inferential(self, records: list[ProposalRecord]):
        results = {}

        row_keys, col_keys, table = self.build_contingency_table(records, "target_tag", "proposal_type")
        chi = self.chisquare_test(table)
        v = self.cramers_v_from_table(table, chi["chi2"])
        results["chi_square_target_by_type"] = {
            "rows": row_keys,
            "cols": col_keys,
            "table": table,
            "chi2": chi["chi2"],
            "dof": chi["dof"],
            "p_value": chi["p_value"],
            "cramers_v": v,
            "scipy_available": SCIPY_AVAILABLE,
        }

        row_keys2, col_keys2, table2 = self.build_contingency_table(records, "split", "proposal_type")
        chi2 = self.chisquare_test(table2)
        v2 = self.cramers_v_from_table(table2, chi2["chi2"])
        results["chi_square_split_by_type"] = {
            "rows": row_keys2,
            "cols": col_keys2,
            "table": table2,
            "chi2": chi2["chi2"],
            "dof": chi2["dof"],
            "p_value": chi2["p_value"],
            "cramers_v": v2,
            "scipy_available": SCIPY_AVAILABLE,
        }

        row_keys3, col_keys3, table3 = self.build_contingency_table(records, "proposal_field", "proposal_type")
        chi3 = self.chisquare_test(table3)
        v3 = self.cramers_v_from_table(table3, chi3["chi2"])
        results["chi_square_field_by_type"] = {
            "rows": row_keys3,
            "cols": col_keys3,
            "table": table3,
            "chi2": chi3["chi2"],
            "dof": chi3["dof"],
            "p_value": chi3["p_value"],
            "cramers_v": v3,
            "scipy_available": SCIPY_AVAILABLE,
        }

        per_target_counts = defaultdict(lambda: {"add": 0, "non_add": 0})
        for r in records:
            if r.proposal_type == "add":
                per_target_counts[r.target_tag]["add"] += 1
            else:
                per_target_counts[r.target_tag]["non_add"] += 1

        targets = sorted(per_target_counts.keys())
        pairwise = []
        pvals = []

        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                ti = targets[i]
                tj = targets[j]

                a = per_target_counts[ti]["add"]
                b = per_target_counts[ti]["non_add"]
                c = per_target_counts[tj]["add"]
                d = per_target_counts[tj]["non_add"]

                ft = self.fisher_test_2x2(a, b, c, d)
                p = ft["p_value"]

                pairwise.append(
                    {
                        "target_a": ti,
                        "target_b": tj,
                        "a_add": int(a),
                        "a_non_add": int(b),
                        "b_add": int(c),
                        "b_non_add": int(d),
                        "odds_ratio": ft["odds_ratio"],
                        "p_value": ft["p_value"],
                    }
                )
                pvals.append(float(p))

        if len(pairwise) > 0:
            qvals = self.bh_fdr(pvals)
            for k in range(len(pairwise)):
                pairwise[k]["q_value_fdr_bh"] = float(qvals[k])

        results["pairwise_fisher_add_share_by_target"] = {
            "comparisons": pairwise,
            "scipy_available": SCIPY_AVAILABLE,
        }

        return results

    # ---------- bootstrap ----------
    def group_records_by_sample(self, records: list[ProposalRecord]):
        grouped = {}
        for r in records:
            key = (r.target_tag, r.split, r.image_id)
            if key in grouped:
                grouped[key].append(r)
            else:
                grouped[key] = [r]
        return grouped

    def compute_add_share_over_samples(self, sample_records_list: list):
        total = 0
        add_cnt = 0
        for recs in sample_records_list:
            for r in recs:
                total += 1
                if r.proposal_type == "add":
                    add_cnt += 1
        if total == 0:
            return 0.0
        return float(add_cnt) / float(total)

    def compute_has_any_proposal_rate_over_samples(self, sample_records_list: list):
        n = len(sample_records_list)
        if n == 0:
            return 0.0
        has = 0
        for recs in sample_records_list:
            if len(recs) > 0:
                has += 1
        return float(has) / float(n)

    def bootstrap_ci_for_share(self, samples: list, share_func_name: str, iters: int, seed: int):
        rnd = random.Random(seed)
        n = len(samples)
        if n == 0:
            return {"mean": None, "ci95_low": None, "ci95_high": None}

        values = []
        for _ in range(iters):
            picked = [samples[rnd.randrange(n)] for _ in range(n)]
            if share_func_name == "add_share":
                share = self.compute_add_share_over_samples(picked)
            else:
                share = self.compute_has_any_proposal_rate_over_samples(picked)
            values.append(float(share))

        values.sort()
        mean = sum(values) / float(len(values))
        low_idx = int(0.025 * (len(values) - 1))
        high_idx = int(0.975 * (len(values) - 1))

        return {
            "mean": float(mean),
            "ci95_low": float(values[low_idx]),
            "ci95_high": float(values[high_idx]),
        }
