# worker/labeling.py

from config.config import Config


from utils.data_utils import get_data
from utils.agent_utils import get_agent


# worker/labeling.py


def run_labeling(config: Config):
    agent = get_agent(config.agent)(config)
    datasets = get_data(config.datasets_dir, config.dataset)

    for split, paths in datasets.items():
        print(f"[{split}] Number of images: {len(paths)}")
        if not paths:
            print(f"[{split}] No images to label. Skipping.")
            continue

        agent.labeling(split, paths)
        print(f"[{split}] Labeling completed.", "-" * 30)

    print("-" * 30)
    print("All labeling tasks completed.")
