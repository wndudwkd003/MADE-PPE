# config/config.py


from dataclasses import dataclass, field

from params.params import (
    DatasetEnum,
    ModelEnum,
    DoModeEnum,
    AgentEnum,
    TestModeEnum
)

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
    clustering_mode: str = "manual"
    clustering_k: int = 8          # manual 모드일 때 사용할 클러스터 개수

    # LABELING or EVALUATION or ANALYSIS
    do_mode: DoModeEnum = DoModeEnum.EVALUATION

    dataset: DatasetEnum = DatasetEnum.SCP300
    model: ModelEnum = ModelEnum.GPT5_MINI

    agent: AgentEnum = AgentEnum.MADE

    workers: int = 30

    runs: str = "runs"
    datasets_dir: str = "datasets"

    seed: int = 42

    api_key: str = "config/api_keys.json"
    max_todo: int = 300  # -1 for all

    explicit_target_tag: int = 3

    temperature: float = 0.7
    top_p: float = 0.9
    max_output_tokens: int = 4096

    role_sequence: list[tuple[RoleEnum, int]] = field(
        default_factory=lambda: ROLE_SEQUENCE
    )

    retry_times: int = 5

    cache_dir: str = "runs/_cache_resized"


    # 테스트 모드
    """
      MADE-PPE:
        - 일관성: 여러 타깃으로 결과물이 일정하게 나오는지
        - 정합성: 만들어진 구조를 사람이 직접 정성 평가

      MADE-Bench:
        - 정확성: 5가지 라벨을 이미지와 직접 비교함 CLIP과 같은 모델 활용

    """
    device: str = "cuda:1"
    sentence_emb_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    blip_model: str = "Salesforce/blip-itm-base-coco"
    gme_model: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"


    test_targets: list[int] = field(
        default_factory=lambda: [1,2,3]
    )

    test_mode: TestModeEnum = TestModeEnum.MADE_BENCH # MADE_PPE or MADE_BENCH

    test_keys: list[str] = field(
        default_factory=lambda: [
            "work_environment",
            "hazards",
            "required_ppe",
            "wearing",
            "improper_wearing",
        ]
    )

    eval_runs: str = "runs_eval"


    top_k: int = 20
