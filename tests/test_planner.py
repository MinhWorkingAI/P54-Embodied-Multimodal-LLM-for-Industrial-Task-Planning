import pytest

from llm_backend.schema import ParsedInstruction, ActionType, ConfidenceLevel
from simulation_backend.action_schema import Position
from task_planner.planner import TaskPlanner
from task_planner.schema import PlannerInput, PlannerOutput, PrimitiveActionType
from task_planner.safety import BoundaryViolation, validate_bounds


def _mock_scene():
    return {
        "objects": [
            {"label": "red block", "position": (0.2, 0.0, 0.05)},
            {"label": "left tray", "position": (0.6, 0.2, 0.05)},
        ]
    }


def test_pick_and_place_decomposition():
    parsed = ParsedInstruction(
        action=ActionType.PICK,
        object_target="red block",
        destination="left tray",
        confidence=ConfidenceLevel.HIGH,
        raw_instruction="pick and place the red block in the left tray",
    )
    planner_input = PlannerInput(parsed_instruction=parsed, scene=_mock_scene())
    planner = TaskPlanner()

    output = planner.generate_primitive_plan(planner_input)

    assert isinstance(output, PlannerOutput)
    action_types = [step.action for step in output.steps]
    assert action_types == [
        PrimitiveActionType.APPROACH,
        PrimitiveActionType.LOWER,
        PrimitiveActionType.GRASP,
        PrimitiveActionType.LIFT,
        PrimitiveActionType.MOVE,
        PrimitiveActionType.RELEASE,
    ]


def test_validate_bounds_blocks_out_of_bounds():
    with pytest.raises(BoundaryViolation):
        validate_bounds(Position(x=1.5, y=0.0, z=0.1))


def test_output_schema_format():
    parsed = ParsedInstruction(
        action=ActionType.PICK,
        object_target="red block",
        destination="left tray",
        confidence=ConfidenceLevel.HIGH,
        raw_instruction="pick and place the red block",
    )
    planner_input = PlannerInput(parsed_instruction=parsed, scene=_mock_scene())
    planner = TaskPlanner()

    output = planner.generate_primitive_plan(planner_input)
    dumped = output.model_dump()

    assert set(dumped.keys()) == {"instruction", "steps", "task_id"}
    assert set(dumped["steps"][0].keys()) == {
        "action",
        "target_object",
        "target_position",
        "parameters",
        "description",
    }
