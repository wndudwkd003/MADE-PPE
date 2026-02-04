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
