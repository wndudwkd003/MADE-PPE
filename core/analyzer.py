import os
import json
import warnings
import re
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from scipy.stats import chi2_contingency

# 콘솔 경고 및 시스템 로그 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config.config import Config
from utils.eval_utils import get_test_targets, get_all_samples, get_image_id, get_cached_image_path

@dataclass
class ProposalRecord:
    target_tag: str
    split: str
    image_id: str
    stage: str
    kind: str
    subject: str
    proposal_type: str
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

        # 폴더 구조 정의
        self.plot_dir = self.save_dir / "plots"
        self.desc_dir = self.plot_dir / "descriptive"
        self.sem_dir = self.plot_dir / "semantic"
        self.wc_dir = self.plot_dir / "wordcloud"
        self.wc_global_dir = self.wc_dir / "global"
        self.wc_split_dir = self.wc_dir / "split"

        # 정성적 평가 샘플 저장 폴더 및 이미지 하위 폴더
        self.gen_proposed_dir = self.save_dir / "gen_proposed"
        self.gen_proposed_images_dir = self.gen_proposed_dir / "images"

        for d in [self.desc_dir, self.sem_dir, self.wc_global_dir, self.wc_split_dir,
                  self.gen_proposed_dir, self.gen_proposed_images_dir]:
            d.mkdir(parents=True, exist_ok=True)

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
        total_samples_processed = 0
        split_samples_processed = defaultdict(int)
        samples_with_proposals_global = set()
        samples_with_proposals_by_split = defaultdict(set)
        copied_samples = set()

        for path in target_paths:
            target_tag = path.name # 예: MADE1, MADE2...
            samples_by_split = get_all_samples(path)
            for split, samples in samples_by_split.items():
                for sample in samples:
                    total_samples_processed += 1
                    split_samples_processed[split] += 1
                    recs = self.extract_records(sample, target_tag, split)
                    if recs:
                        all_records.extend(recs)
                        img_id = recs[0].image_id
                        sample_key = (target_tag, img_id)
                        samples_with_proposals_global.add(sample_key)
                        samples_with_proposals_by_split[split].add(sample_key)

                        # [중요] 고유한 저장을 위해 target_tag 전달
                        self.save_qualitative_sample(sample, split, target_tag, copied_samples)

        if not all_records:
            print("[Analysis] No proposal records found.")
            return

        image_stats = {
            "total": {
                "samples_with_proposals": len(samples_with_proposals_global),
                "total_samples_processed": total_samples_processed,
                "ratio": round(len(samples_with_proposals_global) / total_samples_processed, 4) if total_samples_processed > 0 else 0
            }
        }
        for split, total_s in split_samples_processed.items():
            count_with_p = len(samples_with_proposals_by_split[split])
            image_stats[split] = {
                "samples_with_proposals": count_with_p,
                "total_samples_processed": total_s,
                "ratio": round(count_with_p / total_s, 4) if total_s > 0 else 0
            }

        records_by_split = defaultdict(list)
        for r in all_records:
            records_by_split[r.split].append(r)

        split_analysis_results = {}
        for split, split_recs in records_by_split.items():
            desc = self.perform_descriptive_analysis(split_recs)
            sem, sem_embeddings = self.perform_semantic_analysis(split_recs)
            self.visualize_descriptive(desc, split)
            self.visualize_semantic_all(sem, sem_embeddings, split)

            split_wc_path = self.wc_split_dir / split
            split_wc_path.mkdir(parents=True, exist_ok=True)
            self.generate_wordclouds(split_recs, split, split_wc_path)
            split_analysis_results[split] = {"descriptive": desc, "semantic": sem}

        global_desc = self.perform_descriptive_analysis(all_records)
        global_sem, _ = self.perform_semantic_analysis(all_records)
        self.generate_wordclouds(all_records, "GLOBAL", self.wc_global_dir)

        final_meta = {
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "total_proposals": len(all_records),
            "image_sample_per_total_proposal": image_stats,
            "splits": list(records_by_split.keys())
        }

        final_report = {
            "meta": final_meta,
            "global_analysis": {"descriptive": global_desc, "semantic": global_sem},
            "split_analysis": split_analysis_results,
            "inferential_analysis": self.perform_inferential_analysis(all_records)
        }

        with open(self.save_dir / "analysis_report.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=4, ensure_ascii=False)
        with open(self.save_dir / "global_summary.json", "w", encoding="utf-8") as f:
            json.dump({"meta": final_meta, "summary": {"descriptive": global_desc, "semantic": global_sem}}, f, indent=4, ensure_ascii=False)

        print(f"[Analysis] Results saved to: {self.save_dir}")
        return self.save_dir

    def perform_descriptive_analysis(self, records: List[ProposalRecord]):
        total_count = len(records)
        label_res = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        map_wh = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        map_whp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for r in records:
            # ref에 tag를 포함하여 고유하게 식별 가능하도록 수정
            ref_str = f"{r.split}_{r.target_tag}_{r.image_id}_proposed"

            if r.kind == "label":
                label_res[r.subject][r.proposal_type][(r.target_from, r.target_to)].append(ref_str)
            elif r.kind == "mapping":
                if r.subject == "we_to_hazard":
                    key = r.scope_work_environment or "(missing_we)"
                    map_wh[key][r.proposal_type][(r.target_from, r.target_to)].append(ref_str)
                elif r.subject == "we_hazard_to_ppe":
                    key = f"{r.scope_work_environment or 'None'} || {r.scope_hazard or 'None'}"
                    map_whp[key][r.proposal_type][(r.target_from, r.target_to)].append(ref_str)

        return {
            "label": self.format_topk_with_refs(label_res, total_count),
            "mapping": {
                "we_to_hazard": self.format_topk_with_refs(map_wh, total_count),
                "we_hazard_to_ppe": self.format_topk_with_refs(map_whp, total_count),
            },
        }

    def format_topk_with_refs(self, grouped_data, total_proposals: int):
        out = {}
        for group_key, type_map in grouped_data.items():
            out[group_key] = {}
            for p_type, transitions in type_map.items():
                sorted_items = sorted(transitions.items(), key=lambda x: len(x[1]), reverse=True)[:self.config.top_k]
                rows = []
                for (f, t), refs in sorted_items:
                    rows.append({
                        "from": f, "to": t,
                        "count": len(refs),
                        "percentage": round((len(refs) / total_proposals) * 100, 4),
                        "ref": sorted(list(set(refs))) # 정렬된 고유 참조 목록
                    })
                out[group_key][p_type] = rows
        return out

    def save_qualitative_sample(self, sample: dict, split: str, target_tag: str, copied_samples: set):
        if "image" not in sample: return
        original_path = sample["image"]
        image_id = get_image_id(original_path)

        # 고유 키: split + tag + image_id
        save_key = f"{split}_{target_tag}_{image_id}"
        if save_key in copied_samples: return

        # 1. JSON 저장: {split}_{tag}_{image_id}_proposed.json
        sample_out_path = self.gen_proposed_dir / f"{save_key}_proposed.json"
        with open(sample_out_path, "w", encoding="utf-8") as f:
            json.dump(sample, f, indent=4, ensure_ascii=False)

        # 2. 이미지 복사: images/{split}_{tag}_{image_id}.extension
        dataset_root = Path("datasets") / self.config.dataset.value
        cached_path = Path(get_cached_image_path(original_path, dataset_root))
        if not cached_path.exists():
            cached_path = dataset_root / "images" / Path(original_path).name

        if cached_path.exists():
            # 확장자 유지하며 이름 변경
            dst_image_name = f"{save_key}{cached_path.suffix}"
            shutil.copy(cached_path, self.gen_proposed_images_dir / dst_image_name)

        copied_samples.add(save_key)

    def generate_wordclouds(self, records: List[ProposalRecord], prefix: str, save_path: Path):
        code_pattern = re.compile(r'\bW\d+(_| )?', re.IGNORECASE)
        def clean_label(text: str):
            if not text or text.upper() == "EMPTY": return ""
            return code_pattern.sub('', text).replace("_", " ").strip().upper()

        global_labels = []
        grouped_labels = defaultdict(list)
        for r in records:
            f, t = clean_label(r.target_from), clean_label(r.target_to)
            if f: global_labels.append(f); grouped_labels[(r.stage, r.kind)].append(f)
            if t: global_labels.append(t); grouped_labels[(r.stage, r.kind)].append(t)

        if global_labels:
            self._create_wc_file(" ".join(global_labels), f"{prefix} | COMBINED TARGETS", save_path / f"{prefix.lower()}_combined_targets.png")
        for (stg, knd), lbs in grouped_labels.items():
            self._create_wc_file(" ".join(lbs), f"{prefix} | {stg} ({knd})", save_path / f"{prefix.lower()}_{stg}_{knd}_targets.png")

    def _create_wc_file(self, text: str, title: str, full_path: Path):
        if not text.strip(): return
        wc = WordCloud(width=1000, height=600, background_color='white', colormap='plasma', collocations=False, min_font_size=12).generate(text)
        plt.figure(figsize=(15, 8)); plt.imshow(wc, interpolation='bilinear'); plt.axis('off'); plt.title(title, fontsize=20, pad=20)
        plt.savefig(full_path, bbox_inches='tight'); plt.close()

    def visualize_descriptive(self, desc_results: dict, split: str):
        self._plot_block(desc_results["label"], split, "label")
        self._plot_block(desc_results["mapping"]["we_to_hazard"], split, "mapping_we_to_hazard")
        self._plot_block(desc_results["mapping"]["we_hazard_to_ppe"], split, "mapping_we_hazard_to_ppe")

    def _plot_block(self, block: dict, split: str, prefix: str):
        for group_key, types in block.items():
            for p_type, trans in types.items():
                if not trans: continue
                plt.figure(figsize=(10, 6))
                labels = [f"{t['from'][:15]}->{t['to'][:15]}" for t in trans]
                counts = [t["count"] for t in trans]
                sns.barplot(x=counts, y=labels, hue=labels, palette="magma", legend=False)
                plt.title(f"[{split.upper()}] {prefix} | {group_key} | {p_type.upper()}")
                plt.tight_layout()
                safe_key = str(group_key).replace(" ", "_").replace("/", "_").replace("|", "")
                plt.savefig(self.desc_dir / f"{split}_{prefix}_{safe_key}_{p_type}.png"); plt.close()

    def extract_records(self, sample: dict, target_tag: str, split: str) -> List[ProposalRecord]:
        final_state = sample.get("final_state", {})
        # 이미지 ID 추출 (에러 방지를 위해 방어적 코드 추가)
        image_path = sample.get("image", "")
        image_id = get_image_id(image_path) if image_path else "unknown"

        stage_keys = ["work_environment", "hazard", "compliance"]
        records = []

        for stage in stage_keys:
            # 1. [구버전 호환] stage_outputs 내부에 있는 경우
            # 예: final_state["stage_outputs"]["work_environment"]["proposals"]
            stage_out_nested = final_state.get("stage_outputs", {}).get(stage)
            proposals_nested = stage_out_nested.get("proposals", []) if stage_out_nested else []

            # 2. [신버전] final_state 바로 아래에 있는 경우 (현재 데이터)
            # 예: final_state["proposals_work_environment"]
            proposals_flat = final_state.get(f"proposals_{stage}", [])

            # 두 리스트를 합침 (둘 중 하나만 있어도 동작, 둘 다 있어도 누락 없음)
            combined_proposals = list(proposals_nested) + list(proposals_flat)

            if not combined_proposals:
                continue

            for p in combined_proposals:
                # flag가 true인 것만 처리
                if p.get("flag") is not True:
                    continue

                scope = p.get("scope", {})

                # 데이터 추가
                records.append(ProposalRecord(
                    target_tag=target_tag,
                    split=split,
                    image_id=image_id,
                    stage=stage,
                    kind=p.get("kind", ""),
                    subject=p.get("subject", ""),
                    proposal_type=p.get("type", ""),
                    target_from=str(p.get("target_from", "")).strip() or "EMPTY",
                    target_to=str(p.get("target_to", "")).strip(),
                    proposal_text=p.get("proposal", ""),
                    scope_work_environment=str(scope.get("work_environment", "")).strip(),
                    scope_hazard=str(scope.get("hazard", "")).strip()
                ))

        return records

    def perform_semantic_analysis(self, records: List[ProposalRecord], max_k: int = 10) -> Tuple[Dict, Dict]:
        # 1. 데이터를 Kind별로 분리
        label_groups = defaultdict(list)
        mapping_groups = defaultdict(list)

        for r in records:
            if r.kind == "label":
                label_groups[r.subject].append(r)
            elif r.kind == "mapping":
                mapping_groups[r.subject].append(r)

        analysis_results = {}
        embeddings_by_subject = {}

        # 클러스터링 모드 설정
        mode = getattr(self.config, "clustering_mode", "auto").lower()
        manual_k = getattr(self.config, "clustering_k", 5)

        # -------------------------------------------------------
        # Phase 1: Label Clustering (엔티티별 대표 용어 사전 생성)
        # -------------------------------------------------------
        # 각 subject(예: work_environment, hazard)별로 label_to_rep(매핑 테이블)을 생성합니다.

        for subject, s_recs in label_groups.items():
            all_labels = {r.target_to.upper() for r in s_recs if r.target_to} | \
                         {r.target_from.upper() for r in s_recs if r.target_from != "EMPTY"}
            label_list = sorted(list(all_labels))
            n_samples = len(label_list)

            if n_samples < 3:
                continue

            # 임베딩 생성
            embeddings = self.embed_model.encode([l.replace("_", " ").lower() for l in label_list])
            embeddings_by_subject[subject] = embeddings

            best_k = 2
            max_s = -1.0

            # K 결정 로직 (Auto vs Manual)
            if mode == "manual":
                target_k = min(manual_k, n_samples - 1)
                best_k = max(2, target_k)
            else:
                k_range = range(2, min(max_k, n_samples))
                for k in k_range:
                    km = KMeans(n_clusters=k, n_init="auto", random_state=self.config.seed).fit(embeddings)
                    s = silhouette_score(embeddings, km.labels_)
                    if s > max_s:
                        max_s, best_k = s, k

            # 최종 클러스터링 수행
            km = KMeans(n_clusters=best_k, n_init="auto", random_state=self.config.seed).fit(embeddings)

            # 대표 라벨 선정
            closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, embeddings)
            cluster_to_rep = {i: label_list[idx].upper() for i, idx in enumerate(closest)}

            # [핵심] 원본 라벨 -> 대표 라벨 변환 딕셔너리
            label_to_rep = {l.upper(): cluster_to_rep[km.labels_[idx]] for idx, l in enumerate(label_list)}

            # Label 통계 집계 (기존 로직)
            semantic_types = defaultdict(Counter)
            for r in s_recs:
                s_from = label_to_rep.get(r.target_from.upper(), "EMPTY") if r.target_from != "EMPTY" else "EMPTY"
                s_to = label_to_rep.get(r.target_to.upper(), r.target_to.upper())
                semantic_types[r.proposal_type][(s_from, s_to)] += 1

            type_analysis = {
                pt: [{"from_merged": f, "to_merged": t, "count": c, "pct_within_type": round((c/sum(counts.values()))*100, 2)}
                     for (f, t), c in counts.most_common(10)]
                for pt, counts in semantic_types.items()
            }

            analysis_results[subject] = {
                "kind": "label",
                "clustering_mode": mode,
                "optimal_k": best_k,
                "silhouette_score": float(max_s) if mode == "auto" else None,
                "type_semantic_transitions": type_analysis,
                "label_to_rep": label_to_rep,
                "reps": cluster_to_rep
            }

        # -------------------------------------------------------
        # Phase 2: Mapping Aggregation (병합 기준 적용하여 매핑 분석)
        # -------------------------------------------------------
        # Label 분석에서 생성된 label_to_rep 맵을 사용하여 Mapping 데이터를 변환/집계합니다.

        # 매핑 관계 정의: Mapping Subject -> (Source Subject, Target Subject)
        # 예: we_to_hazard는 work_environment에서 hazard로 가는 매핑임
        mapping_relations = {
            "we_to_hazard": ("work_environment", "hazard"),
            "we_hazard_to_ppe": ("hazard", "compliance"), # 데이터셋에 따라 'compliance' 혹은 'ppe' 확인 필요
            # 필요에 따라 추가
        }

        for map_subject, m_recs in mapping_groups.items():
            # 정의된 관계가 없으면 스킵 (혹은 기본 처리)
            if map_subject not in mapping_relations:
                continue

            src_entity, tgt_entity = mapping_relations[map_subject]

            # Phase 1에서 생성된 맵 가져오기 (없으면 빈 딕셔너리 -> 변환 없이 원본 사용됨)
            src_map = analysis_results.get(src_entity, {}).get("label_to_rep", {})
            tgt_map = analysis_results.get(tgt_entity, {}).get("label_to_rep", {})

            # 매핑 데이터 변환 및 카운팅
            merged_counts = defaultdict(Counter) # type -> (from, to) -> count

            for r in m_recs:
                # 1. Source 변환
                raw_from = r.target_from.upper()
                # 매핑 테이블에 있으면 변환, 없으면 원본 유지 (EMPTY 처리 포함)
                merged_from = src_map.get(raw_from, raw_from) if raw_from != "EMPTY" else "EMPTY"

                # 2. Target 변환
                raw_to = r.target_to.upper()
                merged_to = tgt_map.get(raw_to, raw_to)

                merged_counts[r.proposal_type][(merged_from, merged_to)] += 1

            # 통계 포맷팅
            mapping_analysis = {
                pt: [{"from_merged": f, "to_merged": t, "count": c,
                      "pct_within_type": round((c/sum(counts.values()))*100, 2)}
                     for (f, t), c in counts.most_common(15)] # Mapping은 조합이 많으므로 Top 15
                for pt, counts in merged_counts.items()
            }

            # 결과 저장 (mapping은 클러스터링 정보 없이 통계만 저장)
            analysis_results[map_subject] = {
                "kind": "mapping",
                "source_entity": src_entity,
                "target_entity": tgt_entity,
                "type_semantic_transitions": mapping_analysis
            }

        return analysis_results, embeddings_by_subject


    def perform_inferential_analysis(self, all_records: List[ProposalRecord]):
        df = pd.DataFrame([asdict(r) for r in all_records])
        inf = {}
        for (kind, subject), gdf in df.groupby(["kind", "subject"]):
            ct = pd.crosstab(gdf["split"], gdf["proposal_type"])
            if ct.size > 1:
                chi2, p, _, _ = chi2_contingency(ct)
                inf[f"{kind}__{subject}"] = {"chi2": float(chi2), "p_value": float(p), "distribution": ct.to_dict()}
        return inf

    def visualize_semantic_all(self, sem_results: dict, sem_embeddings: dict, split: str):
        for subj, data in sem_results.items():
            if subj in sem_embeddings: self.visualize_embedding_space(subj, data, sem_embeddings[subj], split)

    def visualize_embedding_space(self, field: str, data: dict, embeddings: np.ndarray, split: str):
        n = embeddings.shape[0]
        labels_in_order = sorted(data["label_to_rep"].keys())
        cluster_labels = [next(i for i, r in data["reps"].items() if r == data["label_to_rep"][l]) for l in labels_in_order]
        reduced = TSNE(n_components=2, perplexity=min(30, n-1), random_state=self.config.seed, init='pca', learning_rate='auto').fit_transform(embeddings)
        plt.figure(figsize=(10, 8)); plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap='tab10', s=100)
        for i, rep in data["reps"].items():
            mask = np.array(cluster_labels) == i
            if mask.any(): plt.annotate(rep, reduced[mask].mean(axis=0), fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.6))
        plt.title(f"[{split.upper()}] {field} Semantic Space"); plt.savefig(self.sem_dir / f"{split}_embedding_{field}.png"); plt.close()
