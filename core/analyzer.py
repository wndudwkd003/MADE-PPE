import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

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
    proposal_field: str
    proposal_type: str
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

        all_records: list[ProposalRecord] = []
        for path in target_paths:
            target_tag = path.name
            samples_by_split = get_all_samples(path)
            for split, samples in samples_by_split.items():
                for sample in samples:
                    recs = self.extract_records(sample, target_tag, split)
                    all_records.extend(recs)

        if not all_records: return

        # 스플릿별 데이터 분리
        records_by_split = defaultdict(list)
        for r in all_records:
            records_by_split[r.split].append(r)

        # 1. 분석 수행 (스플릿별)
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
            sem = self.perform_semantic_analysis(split_recs)

            # 시각화 (파일명에 split 포함)
            self.visualize_descriptive(desc, split)
            self.visualize_semantic_all(sem, split)
            self.visualize_wordcloud(split_recs, split)

            final_report["split_analysis"][split] = {
                "descriptive": desc,
                "semantic": sem
            }

        # 2. 전체 데이터 기반 추론 통계 (Split 간의 차이를 분석해야 하므로 전체 사용)
        final_report["inferential_analysis"] = self.perform_inferential_analysis(all_records)
        final_report["meta"]["finished_at"] = datetime.now().isoformat(timespec="seconds")

        json_path = self.save_dir / "analysis_report.json"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=4, ensure_ascii=False)

        return json_path



    def extract_records(self, sample: dict, target_tag: str, split: str) -> list[ProposalRecord]:
        final_state = sample["final_state"]
        image_id = get_image_id(sample["image"])
        proposals = self.get_proposal(final_state)

        records = []
        for pk, pv in proposals:
            field = pk[len("proposal_") :]
            p_type = pv["type"]
            t_from = "EMPTY" if p_type == "add" else (pv.get("target_from") or str(final_state.get(field, "")))

            records.append(ProposalRecord(
                target_tag=target_tag, split=split, image_id=image_id,
                proposal_field=field, proposal_type=p_type,
                target_from=t_from, target_to=pv["target_to"],
                proposal_text=pv["proposal"]
            ))
        return records

    def get_proposal(self, final_state: dict[str, Any]):
        return [(k, v) for k, v in final_state.items() if k.startswith("proposal_") and v.get("flag") is True]

    def perform_descriptive_analysis(self, records: list[ProposalRecord]):
        results = defaultdict(lambda: defaultdict(Counter))
        for r in records:
            results[r.proposal_field][r.proposal_type][(r.target_from, r.target_to)] += 1

        formatted = {}
        for field, types in results.items():
            formatted[field] = {}
            for p_type, counts in types.items():
                formatted[field][p_type] = [
                    {"from": f, "to": t, "count": c}
                    for (f, t), c in counts.most_common(self.config.top_k)
                ]
        return formatted

    def perform_semantic_analysis(self, records: list[ProposalRecord], max_k: int = 10):
        field_groups = defaultdict(list)
        for r in records:
            field_groups[r.proposal_field].append(r)

        analysis_by_field = {}
        for field, f_recs in field_groups.items():
            labels = list(set(r.target_to.replace('_', ' ').lower() for r in f_recs if r.target_to) |
                          set(r.target_from.replace('_', ' ').lower() for r in f_recs if r.target_from != "EMPTY"))

            if len(labels) < 3: continue
            embeddings = self.embed_model.encode(labels)

            best_k, max_s, scores = 2, -1, []
            k_range = range(2, min(max_k, len(labels)))
            for k in k_range:
                km = KMeans(n_clusters=k, n_init='auto', random_state=self.config.seed).fit(embeddings)
                s = silhouette_score(embeddings, km.labels_)
                scores.append(s)
                if s > max_s: max_s, best_k = s, k

            km = KMeans(n_clusters=best_k, n_init='auto', random_state=self.config.seed).fit(embeddings)
            closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, embeddings)
            cluster_to_rep = {i: labels[idx].upper().replace(' ', '_') for i, idx in enumerate(closest)}

            group_details = {cluster_to_rep[i]: {"count": 0, "members": []} for i in range(best_k)}
            for idx, label_idx in enumerate(km.labels_):
                group_details[cluster_to_rep[label_idx]]["members"].append(labels[idx].upper().replace(' ', '_'))

            for r in f_recs:
                target_clean = r.target_to.replace('_', ' ').upper()
                for rep, info in group_details.items():
                    if target_clean in info["members"]:
                        info["count"] += 1
                        break

            analysis_by_field[field] = {
                "optimal_k": best_k, "silhouette_score": float(max_s),
                "k_search": {"ks": list(k_range), "scores": [float(s) for s in scores]},
                "groups": group_details, "embeddings": embeddings.tolist(),
                "cluster_labels": km.labels_.tolist(), "reps": cluster_to_rep
            }
        return analysis_by_field

    def perform_inferential_analysis(self, all_records: list[ProposalRecord]):
        df = pd.DataFrame([asdict(r) for r in all_records])
        inf = {}
        for field in df['proposal_field'].unique():
            f_df = df[df['proposal_field'] == field]
            ct = pd.crosstab(f_df['split'], f_df['proposal_type'])
            if ct.size > 1:
                chi2, p, _, _ = chi2_contingency(ct)
                inf[f"{field}_split_vs_type"] = {"chi2": chi2, "p_value": p, "distribution": ct.to_dict()}
        return inf

    def visualize_descriptive(self, desc_results: dict, split: str):
        for field, types in desc_results.items():
            for p_type, trans in types.items():
                if not trans: continue
                plt.figure(figsize=(10, 6))
                labels = [f"{t['from']} -> {t['to']}" for t in trans]
                counts = [t['count'] for t in trans]
                sns.barplot(x=counts, y=labels, palette="magma")
                plt.title(f"[{split.upper()}] {field} - {p_type.upper()}")
                plt.tight_layout()
                plt.savefig(self.save_dir / f"{split}_trans_{field}_{p_type}.png")
                plt.close()

    def visualize_semantic_all(self, sem_results: dict, split: str):
        for field, data in sem_results.items():
            plt.figure(figsize=(6, 4))
            plt.plot(data["k_search"]["ks"], data["k_search"]["scores"], marker='o')
            plt.title(f"[{split.upper()}] {field} Silhouette")
            plt.savefig(self.save_dir / f"{split}_silhouette_{field}.png")
            plt.close()
            self.visualize_embedding_space(field, data, split)

    def visualize_embedding_space(self, field: str, data: dict, split: str):
        embeddings = np.array(data["embeddings"])
        n = embeddings.shape[0]
        reducer = TSNE(n_components=2, perplexity=min(30, n-1), random_state=self.config.seed, init='pca', learning_rate='auto')
        reduced = reducer.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=data["cluster_labels"], cmap='tab10', s=100)
        for i, rep in data["reps"].items():
            mask = np.array(data["cluster_labels"]) == i
            if not mask.any(): continue
            center = reduced[mask].mean(axis=0)
            plt.annotate(rep, center, fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6))
        plt.title(f"[{split.upper()}] {field} Semantic Space")
        plt.savefig(self.save_dir / f"{split}_embedding_{field}.png")
        plt.close()

    def visualize_wordcloud(self, records: list[ProposalRecord], split: str):
        text = " ".join([r.proposal_text for r in records if r.proposal_text])
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc)
            plt.axis('off')
            plt.title(f"[{split.upper()}] Proposal Reasons")
            plt.savefig(self.save_dir / f"{split}_wordcloud.png")
            plt.close()
