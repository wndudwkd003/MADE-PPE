import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from scipy.stats import chi2_contingency

from config.config import Config
from utils.eval_utils import get_test_targets, get_all_samples, get_image_id

@dataclass
class ProposalRecord:
    target_tag: str
    split: str
    image_id: str

    stage: str          # work_environment | hazard | compliance
    kind: str           # label | mapping
    subject: str        # work_environment|hazard|ppe | we_to_hazard|we_hazard_to_ppe
    proposal_type: str  # add | remove | modify

    target_from: str
    target_to: str
    proposal_text: str

    scope_work_environment: str
    scope_hazard: str


class Analyzer:
    def __init__(self, config: Config):
        self.config = config
        self.embed_model = SentenceTransformer(
            config.sentence_emb_model,
            device=config.device,
        )

        self.save_dir = (
            Path(self.config.eval_runs) /
            "analysis" /
            str(self.config.dataset.name) /
            str(self.config.agent.name)
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('ggplot')

    def run(self):
        started_at = datetime.now().isoformat(timespec="seconds")
        target_paths = get_test_targets(
            agent=self.config.agent,
            dataset=self.config.dataset,
            targets=self.config.test_targets,
            run_dir=self.config.runs,
        )

        all_records: List[ProposalRecord] = []
        for path in target_paths:
            target_tag = path.name
            samples_by_split = get_all_samples(path)
            for split, samples in samples_by_split.items():
                for sample in samples:
                    recs = self.extract_records(sample, target_tag, split)
                    all_records.extend(recs)

        if not all_records: return

        records_by_split = defaultdict(list)
        for r in all_records:
            records_by_split[r.split].append(r)

        final_report = {
            "meta": {
                "started_at": started_at,
                "total_proposals": len(all_records),
                "splits": list(records_by_split.keys())
            },
            "split_analysis": {}
        }

        for split, split_recs in records_by_split.items():
            desc = self.perform_descriptive_analysis(split_recs)
            # 의미론적 분석 수행 (Type별 분석 포함)
            sem_results, sem_embeddings = self.perform_semantic_analysis(split_recs)

            self.visualize_descriptive(desc, split)
            self.visualize_semantic_all(sem_results, sem_embeddings, split)
            self.visualize_wordcloud(split_recs, split)

            final_report["split_analysis"][split] = {
                "descriptive": desc,
                "semantic": sem_results
            }

        final_report["inferential_analysis"] = self.perform_inferential_analysis(all_records)
        final_report["meta"]["finished_at"] = datetime.now().isoformat(timespec="seconds")

        json_path = self.save_dir / "analysis_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=4, ensure_ascii=False)

        return json_path

    def extract_records(self, sample: dict, target_tag: str, split: str) -> List[ProposalRecord]:
        final_state = sample["final_state"]
        image_id = get_image_id(sample["image"])

        stage_keys = ["work_environment", "hazard", "compliance"]

        records = []
        for stage in stage_keys:
            stage_outputs = final_state.get("stage_outputs", {})
            stage_out = stage_outputs.get(stage)
            if not stage_out:
                continue
            proposals = stage_out.get("proposals", [])
            for p in proposals:
                if p["flag"] is not True:
                    continue

                scope = p["scope"]
                scope_we = scope["work_environment"]
                scope_hz = scope["hazard"]

                p_type = p["type"]
                t_from = p["target_from"]
                t_to = p["target_to"]

                if p_type == "add":
                    if t_from == "":
                        t_from_norm = "EMPTY"
                    else:
                        t_from_norm = t_from.strip()
                else:
                    t_from_norm = t_from.strip()

                records.append(
                    ProposalRecord(
                        target_tag=target_tag,
                        split=split,
                        image_id=image_id,

                        stage=stage,
                        kind=p["kind"],
                        subject=p["subject"],
                        proposal_type=p_type,

                        target_from=t_from_norm,
                        target_to=t_to.strip(),
                        proposal_text=p["proposal"],

                        scope_work_environment=scope_we.strip(),
                        scope_hazard=scope_hz.strip(),
                    )
                )

        return records


    def perform_descriptive_analysis(self, records: List[ProposalRecord]):
        label_results = defaultdict(lambda: defaultdict(Counter))
        we_to_hazard_results = defaultdict(lambda: defaultdict(Counter))
        we_hazard_to_ppe_results = defaultdict(lambda: defaultdict(Counter))

        for r in records:
            if r.kind == "label":
                key = r.subject
                label_results[key][r.proposal_type][(r.target_from, r.target_to)] += 1
                continue

            if r.kind == "mapping" and r.subject == "we_to_hazard":
                key = r.scope_work_environment if r.scope_work_environment else "(missing_we)"
                we_to_hazard_results[key][r.proposal_type][(r.target_from, r.target_to)] += 1
                continue

            if r.kind == "mapping" and r.subject == "we_hazard_to_ppe":
                key = (
                    r.scope_work_environment if r.scope_work_environment else "(missing_we)",
                    r.scope_hazard if r.scope_hazard else "(missing_hazard)",
                )
                we_hazard_to_ppe_results[key][r.proposal_type][(r.target_from, r.target_to)] += 1
                continue

        return {
            "label": self.format_topk(label_results, self.config.top_k),
            "mapping": {
                "we_to_hazard": self.format_topk(we_to_hazard_results, self.config.top_k),
                "we_hazard_to_ppe": self.format_topk(we_hazard_to_ppe_results, self.config.top_k),
            },
        }


    def perform_semantic_analysis(self, records: List[ProposalRecord], max_k: int = 10) -> Tuple[Dict, Dict]:
        subject_groups = defaultdict(list)
        for r in records:
            if r.kind != "label":
                continue
            subject_groups[r.subject].append(r)

        analysis_by_subject = {}
        embeddings_by_subject = {}

        for subject, s_recs in subject_groups.items():
            all_labels = set()
            for r in s_recs:
                if r.target_to:
                    all_labels.add(r.target_to.upper())
                if r.target_from and r.target_from != "EMPTY":
                    all_labels.add(r.target_from.upper())

            label_list = sorted(list(all_labels))
            if len(label_list) < 3:
                continue

            clean_labels = [l.replace("_", " ").lower() for l in label_list]
            embeddings = self.embed_model.encode(clean_labels)
            embeddings_by_subject[subject] = embeddings

            best_k, max_s = 2, -1
            k_range = range(2, min(max_k, len(label_list)))
            for k in k_range:
                km = KMeans(n_clusters=k, n_init="auto", random_state=self.config.seed).fit(embeddings)
                s = silhouette_score(embeddings, km.labels_)
                if s > max_s:
                    max_s, best_k = s, k

            km = KMeans(n_clusters=best_k, n_init="auto", random_state=self.config.seed).fit(embeddings)
            closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, embeddings)
            cluster_to_rep = {i: label_list[idx].upper() for i, idx in enumerate(closest)}

            label_to_rep = {}
            for idx, label in enumerate(label_list):
                label_to_rep[label.upper()] = cluster_to_rep[km.labels_[idx]]

            semantic_types = defaultdict(Counter)
            for r in s_recs:
                s_from = "EMPTY"
                if r.target_from != "EMPTY":
                    s_from = label_to_rep[r.target_from.upper()] if r.target_from.upper() in label_to_rep else r.target_from.upper()

                s_to = label_to_rep[r.target_to.upper()] if r.target_to.upper() in label_to_rep else r.target_to.upper()

                semantic_types[r.proposal_type][(s_from, s_to)] += 1

            type_analysis = {}
            for p_type, counts in semantic_types.items():
                type_analysis[p_type] = [
                    {"from_merged": f, "to_merged": t, "count": c}
                    for (f, t), c in counts.most_common(10)
                ]

            analysis_by_subject[subject] = {
                "optimal_k": best_k,
                "silhouette_score": float(max_s),
                "type_semantic_transitions": type_analysis,
                "reps": cluster_to_rep,
                "label_to_rep": label_to_rep,
            }

        return analysis_by_subject, embeddings_by_subject


    def perform_inferential_analysis(self, all_records: List[ProposalRecord]):
        df = pd.DataFrame([asdict(r) for r in all_records])

        inf = {}
        groups = df.groupby(["kind", "subject"])
        for (kind, subject), gdf in groups:
            ct = pd.crosstab(gdf["split"], gdf["proposal_type"])
            if ct.size <= 1:
                continue
            chi2, p, _, _ = chi2_contingency(ct)
            key = f"{kind}__{subject}__split_vs_type"
            inf[key] = {"chi2": float(chi2), "p_value": float(p), "distribution": ct.to_dict()}

        return inf


    def visualize_descriptive(self, desc_results: dict, split: str):
        label_desc = desc_results["label"]
        self._visualize_desc_block(label_desc, split, prefix="label")

        mapping_desc = desc_results["mapping"]

        we_to_hazard_desc = mapping_desc["we_to_hazard"]
        self._visualize_desc_block(we_to_hazard_desc, split, prefix="mapping_we_to_hazard")

        we_hazard_to_ppe_desc = mapping_desc["we_hazard_to_ppe"]
        self._visualize_desc_block(we_hazard_to_ppe_desc, split, prefix="mapping_we_hazard_to_ppe")


    def _visualize_desc_block(self, block: dict, split: str, prefix: str):
        for group_key, types in block.items():
            for p_type, trans in types.items():
                if not trans:
                    continue

                plt.figure(figsize=(10, 6))
                labels = [f"{t['from'][:20]} -> {t['to'][:20]}" for t in trans]
                counts = [t["count"] for t in trans]

                sns.barplot(x=counts, y=labels, palette="magma")
                plt.title(f"[{split.upper()}] {prefix} | {group_key} | {p_type.upper()}")
                plt.tight_layout()
                safe_group = str(group_key).replace(" ", "_").replace("/", "_")
                plt.savefig(self.save_dir / f"{split}_{prefix}_{safe_group}_{p_type}.png")
                plt.close()


    def visualize_semantic_all(self, sem_results: dict, sem_embeddings: dict, split: str):
        for subject, data in sem_results.items():
            if subject not in sem_embeddings:
                continue
            self.visualize_embedding_space(subject, data, sem_embeddings[subject], split)


    def visualize_embedding_space(self, field: str, data: dict, embeddings: np.ndarray, split: str):
        n = embeddings.shape[0]
        # label_to_rep의 순서대로 클러스터 ID 할당
        labels_in_order = sorted(data["label_to_rep"].keys())
        cluster_labels = []
        for l in labels_in_order:
            rep = data["label_to_rep"][l]
            for idx, r_name in data["reps"].items():
                if r_name == rep:
                    cluster_labels.append(idx)
                    break

        reducer = TSNE(n_components=2, perplexity=min(30, n-1), random_state=self.config.seed, init='pca', learning_rate='auto')
        reduced = reducer.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap='tab10', s=100)
        for i, rep in data["reps"].items():
            mask = np.array(cluster_labels) == i
            if not mask.any(): continue
            center = reduced[mask].mean(axis=0)
            plt.annotate(rep, center, fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6))
        plt.title(f"[{split.upper()}] {field} Semantic Space")
        plt.savefig(self.save_dir / f"{split}_embedding_{field}.png")
        plt.close()

    def visualize_wordcloud(self, records: List[ProposalRecord], split: str):
        text = " ".join([r.proposal_text for r in records if r.proposal_text])
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc)
            plt.axis('off')
            plt.title(f"[{split.upper()}] Proposal Reasons")
            plt.savefig(self.save_dir / f"{split}_wordcloud.png")
            plt.close()
