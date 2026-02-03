# worker/labeling.py

from config.config import Config


from utils.data_utils import get_data
from utils.agent_utils import get_agent


def run_labeling(config: Config):
    # MADE / SingleAgent
    agent = get_agent(config.agent)(config)

    # {"train": [...], "valid": [...], "test": [...]}
    datasets = get_data(config.datasets_dir, config.dataset)

    for split, paths in datasets.items():
        print(f"[{split}] Number of images: {len(paths)}")
        if not paths:
            continue

        agent.labeling(config, split, paths)
        print(f"[{split}] Labeling completed.")

    print("-" * 30)
    print("All labeling tasks completed.")
