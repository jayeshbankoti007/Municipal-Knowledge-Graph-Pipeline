from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class PredictionStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    UNCERTAIN = "UNCERTAIN"

class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class VoteValue(str, Enum):
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"
    HELD = "held"

# Entity Models
class Bill(BaseModel):
    """Represents a bill, ordinance, or resolution"""
    id: str = Field(description="Bill identifier (e.g., '25-O-1271')")
    title: str = Field(description="Full title or description of the bill")
    type: Optional[str] = Field(default=None, description="Type: ordinance, resolution, etc.")
    prediction: PredictionStatus = Field(description="Predicted outcome")
    confidence: Confidence = Field(description="Confidence in prediction")
    reasoning: str = Field(description="Brief explanation for prediction")

class Person(BaseModel):
    """Represents a person mentioned in transcript"""
    name: str = Field(description="Full name of person")
    role: Optional[str] = Field(default=None, description="Role or title")
    organization: Optional[str] = Field(default=None, description="Affiliated organization")

class Organization(BaseModel):
    """Represents an organization, department, or company"""
    name: str = Field(description="Name of organization")
    type: Optional[str] = Field(default=None, description="Type: department, company, agency, etc.")

class Project(BaseModel):
    """Represents a real estate or infrastructure project"""
    name: str = Field(description="Project name or description")
    type: Optional[str] = Field(default=None, description="Project type: residential, commercial, etc.")
    location: Optional[str] = Field(default=None, description="Address or location")
    amount: Optional[str] = Field(default=None, description="Project budget/value")

class Vote(BaseModel):
    """Represents a vote on a bill"""
    bill_id: str = Field(description="Bill identifier being voted on")
    person: str = Field(description="Name of person voting")
    vote: VoteValue = Field(description="Vote value")

class TranscriptExtraction(BaseModel):
    """Complete extraction from a single transcript"""
    bills: List[Bill] = Field(default_factory=list)
    people: List[Person] = Field(default_factory=list)
    organizations: List[Organization] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    votes: List[Vote] = Field(default_factory=list)


# Resolution Models
# class ResolvedEntity(BaseModel):
#     """A resolved canonical entity with its aliases"""
#     canonical: str = Field(description="Canonical/preferred name")
#     aliases: List[str] = Field(default_factory=list, description="Alternative names found")

# class EntityResolution(BaseModel):
#     """Complete entity resolution results"""
#     organizations: List[ResolvedEntity] = Field(default_factory=list)
#     bills: List[ResolvedEntity] = Field(default_factory=list)
#     projects: List[ResolvedEntity] = Field(default_factory=list)