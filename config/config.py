# config/config.py


from dataclasses import dataclass, field

from params.params import DatasetEnum, ModelEnum, DoModeEnum, AgentEnum


@dataclass
class Config:
    do_mode: DoModeEnum = DoModeEnum.LABELING
    dataset: DatasetEnum = DatasetEnum.SH17
    model: ModelEnum = ModelEnum.GPT5_MINI

    agent: AgentEnum = AgentEnum.MADE

    able_gpus: list[int] = field(default_factory=lambda: [0, 1, 2])
    workers: int = 10

    runs: str = "runs"
    datasets_dir: str = "datasets"

    seed: int = 42

    api_key: str = "config/api_keys.json"
    max_todo: int = 300  # -1 for all
