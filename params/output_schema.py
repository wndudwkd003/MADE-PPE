# params/output_schema.py

from pydantic import BaseModel, Field


class ProposalOut(BaseModel):
    flag: bool = Field(...)
    type: str = Field(...)
    target_from: str = Field(...)
    target_to: str = Field(...)
    proposal: str = Field(...)


class WorkEnvironmentOut(BaseModel):
    work_environment: str = Field(...)
    reason: str = Field(...)
    proposal: ProposalOut = Field(...)


class HazardOut(BaseModel):
    hazards: list[str] = Field(default_factory=list)
    reason: str = Field(...)
    proposal: ProposalOut = Field(...)


class ComplianceOut(BaseModel):
    required_ppe: list[str] = Field(default_factory=list)
    reason: str = Field(...)
    proposal: ProposalOut = Field(...)


class PPEItem(BaseModel):
    ppe: str = Field(...)
    worn: bool = Field(...)


class WearingOut(BaseModel):
    wearing: list[PPEItem] = Field(default_factory=list)
    reason: str = Field(...)


class ImproperWearingOut(BaseModel):
    improper_wearing: list[PPEItem] = Field(default_factory=list)
    reason: str = Field(...)


class TextOut(BaseModel):
    text: str = Field(...)

class OneShotOut(BaseModel):
    # Stage 1: work_environment
    work_environment: str = Field(...)
    work_environment_reason: str = Field(...)
    proposal_work_environment: ProposalOut = Field(...)

    # Stage 2: hazards
    hazards: list[str] = Field(default_factory=list)
    hazards_reason: str = Field(...)
    proposal_hazard: ProposalOut = Field(...)

    # Stage 3: compliance (required PPE)
    required_ppe: list[str] = Field(default_factory=list)
    required_ppe_reason: str = Field(...)
    proposal_compliance: ProposalOut = Field(...)

    # Stage 4: wearing
    wearing: list[PPEItem] = Field(default_factory=list)
    wearing_reason: str = Field(...)

    # Stage 5: improper_wearing
    improper_wearing: list[PPEItem] = Field(default_factory=list)
    improper_wearing_reason: str = Field(...)