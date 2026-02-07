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
    result_path = analyzer.run()

    print(f"[Analysis] Results saved to: {result_path}")


