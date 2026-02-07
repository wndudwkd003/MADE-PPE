# params/output_schema.py

from pydantic import BaseModel, Field


class ProposalScope(BaseModel):
    work_environment: str = Field("")
    hazard: str = Field("")


class ProposalOut(BaseModel):
    flag: bool = Field(...)

    # 제안 종류 구분
    # - label: 라벨 집합(WorkEnvironment/HazardFactor/PPEItem Enum) 변경
    # - mapping: WORK_ENV_TO_HAZARD_PPE 테이블(구조화) 엔트리 변경
    kind: str = Field(...)

    # kind별 세부 대상
    # kind=label:
    # - work_environment | hazard | ppe
    # kind=mapping:
    # - we_to_hazard | we_hazard_to_ppe
    subject: str = Field(...)

    # add | remove | modify
    type: str = Field(...)

    # 변경 전/후 키
    # - label 제안이면: 라벨 키(예: NEW_HAZARD, NEW_PPE 등)
    # - mapping 제안이면: hazard 키 또는 ppe 키(아래 scope와 결합해서 위치가 결정됨)
    target_from: str = Field(...)
    target_to: str = Field(...)

    proposal: str = Field(...)

    # mapping 제안일 때 위치 정보
    # - subject=we_to_hazard: work_environment 필수, hazard는 target_from/target_to로 표현
    # - subject=we_hazard_to_ppe: work_environment + hazard 필수, ppe는 target_from/target_to로 표현
    scope: ProposalScope = Field(default_factory=ProposalScope)


class WorkEnvironmentOut(BaseModel):
    work_environment: str = Field(...)
    reason: str = Field(...)
    proposals: list[ProposalOut] = Field(default_factory=list)


class HazardOut(BaseModel):
    hazards: list[str] = Field(default_factory=list)
    reason: str = Field(...)
    proposals: list[ProposalOut] = Field(default_factory=list)


class ComplianceOut(BaseModel):
    required_ppe: list[str] = Field(default_factory=list)
    reason: str = Field(...)
    proposals: list[ProposalOut] = Field(default_factory=list)


class PPEStatusItem(BaseModel):
    ppe: str = Field(...)
    worn: bool = Field(...)


class WearingOut(BaseModel):
    wearing: list[PPEStatusItem] = Field(default_factory=list)
    reason: str = Field(...)


class ImproperWearingOut(BaseModel):
    improper_wearing: list[PPEStatusItem] = Field(default_factory=list)
    reason: str = Field(...)


class TextOut(BaseModel):
    text: str = Field(...)


class OneShotOut(BaseModel):
    work_environment: WorkEnvironmentOut = Field(...)
    hazard: HazardOut = Field(...)
    compliance: ComplianceOut = Field(...)
    wearing: WearingOut = Field(...)
    improper_wearing: ImproperWearingOut = Field(...)
