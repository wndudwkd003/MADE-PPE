# worker/analysis.py

from __future__ import annotations

from pathlib import Path

from config.config import Config
from core.analyzer import Analyzer
from utils.json_utils import dump_json
from utils.vis_utils import plot_bar, plot_grouped


def run_analysis(config: Config):
    out_dir = Path(config.eval_runs) / "analysis" / config.dataset.value / config.agent.name
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = Analyzer(config)
    summary = analyzer.run()

    dump_json(
        out_dir / "summary.json",
        {
            "summary": summary,
            "meta": {
                "dataset": config.dataset.name,
                "agent": config.agent.name,
                "targets": config.test_targets,
                "runs": config.runs,
                "eval_runs": config.eval_runs,
                "scipy_available": summary["meta"]["scipy_available"],
            },
        },
    )

    desc = summary["descriptive"]
    grouped = summary["grouped"]
    type_keys = grouped["type_keys"]

    plot_bar(desc["proposal_type_counts"], out_dir / "proposal_type_bar.png", title="Proposal type distribution")
    plot_bar(desc["proposal_field_counts"], out_dir / "proposal_field_bar.png", title="Proposal field distribution")
    plot_bar(desc["top_transitions"], out_dir / "top_transitions_bar.png", title=f"Top transitions (Top-{config.top_k})")
    plot_bar(desc["top_from"], out_dir / "top_from_bar.png", title=f"Top target_from (Top-{config.top_k})")
    plot_bar(desc["top_to"], out_dir / "top_to_bar.png", title=f"Top target_to (Top-{config.top_k})")


    plot_grouped(grouped["by_target_type"], keys=type_keys, out_path=out_dir / "grouped_by_target_type.png", title="Proposal type distribution (by target)")
    plot_grouped(grouped["by_field_type"], keys=type_keys, out_path=out_dir / "grouped_by_field_type.png", title="Proposal type distribution (by field)")

    plot_grouped(
        grouped["by_target_type"],
        keys=type_keys,
        out_path=out_dir / "grouped_by_target_type.png",
        title="Proposal type distribution (by target, counts)",
        normalize=False,
    )

    plot_grouped(
        grouped["by_field_type"],
        keys=type_keys,
        out_path=out_dir / "grouped_by_field_type.png",
        title="Proposal type distribution (by field, counts)",
        normalize=False,
    )

    plot_grouped(
        grouped["by_target_type"],
        keys=type_keys,
        out_path=out_dir / "grouped_by_target_type_ratio.png",
        title="Proposal type distribution (by target, ratio)",
        normalize=True,
    )

    plot_grouped(
        grouped["by_field_type"],
        keys=type_keys,
        out_path=out_dir / "grouped_by_field_type_ratio.png",
        title="Proposal type distribution (by field, ratio)",
        normalize=True,
    )


    print(f"\n[ANALYSIS] Saved summary: {out_dir / 'summary.json'}")
    print(f"[ANALYSIS] Saved plots under: {out_dir}")
