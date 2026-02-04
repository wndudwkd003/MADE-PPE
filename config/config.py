# config/config.py


from dataclasses import dataclass, field

from params.params import DatasetEnum, ModelEnum, DoModeEnum, AgentEnum

from params.prompt_params import RoleEnum


ROLE_SEQUENCE = [
    (RoleEnum.PROPOSER, 1),
    (RoleEnum.REBUTTER, 1),
    (RoleEnum.PROPOSER, 2),
    (RoleEnum.REBUTTER, 2),
    (RoleEnum.JUDGE, 1),
]


@dataclass
class Config:
    do_mode: DoModeEnum = DoModeEnum.LABELING
    dataset: DatasetEnum = DatasetEnum.SH17
    model: ModelEnum = ModelEnum.GPT5_MINI

    agent: AgentEnum = AgentEnum.MADE

    workers: int = 1

    runs: str = "runs"
    datasets_dir: str = "datasets"

    seed: int = 42

    api_key: str = "config/api_keys.json"
    max_todo: int = 300  # -1 for all

    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 2048

    role_sequence: list[tuple[RoleEnum, int]] = field(
        default_factory=lambda: ROLE_SEQUENCE
    )

    retry_times: int = 5


