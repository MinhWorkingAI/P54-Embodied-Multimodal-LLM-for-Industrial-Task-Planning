"""
schema.py
---------
Input/output schema for the task planner.
Defines a stable contract between LLM parsing, scene input, and primitive action output.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from llm_backend.schema import ParsedInstruction
from simulation_backend.action_schema import Position


class PrimitiveActionType(str, Enum):
    APPROACH = "approach"
    LOWER = "lower"
    GRASP = "grasp"
    LIFT = "lift"
    MOVE = "move"
    RELEASE = "release"
    LOCATE = "locate"


class PrimitiveAction(BaseModel):
    action: PrimitiveActionType = Field(description="Primitive action name.")
    target_object: Optional[str] = Field(
        default=None,
        description="Target object for this action, if any.",
    )
    target_position: Optional[Position] = Field(
        default=None,
        description="Target position for this action, if any.",
    )
    parameters: dict = Field(default_factory=dict, description="Extra parameters.")
    description: str = Field(default="", description="Human-readable description.")


class PlannerInput(BaseModel):
    parsed_instruction: ParsedInstruction = Field(
        description="Parsed LLM instruction for planning."
    )
    scene: dict = Field(description="Scene map from the vision module.")


class PlannerOutput(BaseModel):
    instruction: str = Field(description="Original instruction string.")
    steps: list[PrimitiveAction] = Field(description="Ordered list of primitive actions.")
    task_id: Optional[str] = Field(default=None, description="Optional tracker task ID.")
