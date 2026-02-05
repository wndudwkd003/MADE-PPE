# run.py

from params.params import DoModeEnum
from config.config import Config
from utils.seeds import set_seed
from utils.api_utils import set_api_key


def main(config: Config):

    if config.do_mode == DoModeEnum.LABELING:
        from worker.labeling import run_labeling

        run_labeling(config)

    elif config.do_mode == DoModeEnum.EVALUATION:
        from worker.evaluation import run_evaluation

        run_evaluation(config)

    elif config.do_mode == DoModeEnum.ANALYSIS:
        from worker.analysis import run_analysis

        run_analysis(config)

    else:
        print(f"Unknown do_mode: {config.do_mode}")


if __name__ == "__main__":
    # config/config.py
    config = Config()

    # Set seed and API keys
    set_seed(config.seed)
    set_api_key(config.api_key)

    # Run main function
    main(config)
