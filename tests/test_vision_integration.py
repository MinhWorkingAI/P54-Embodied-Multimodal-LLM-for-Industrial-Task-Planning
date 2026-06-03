from pathlib import Path

import pytest

from llm_backend.schema import ActionType, ConfidenceLevel, ParsedInstruction
from llm_backend.tracker import PipelineTracker
from main import run_pipeline
from simulation_backend.action_schema import CommandType
from task_planner.planner import TaskPlanner
from vision_backend.scene_representation import get_current_scene


VISION_OUTPUT = Path(__file__).resolve().parents[1] / "drafts" / "scene_output.json"
EXPECTED_POSITIONS = {
    "red block": [150.0, 110.0],
    "blue tray": [350.0, 235.0],
    "green block": [247.5, 127.5],
}


def _object_positions(scene):
    return {obj["label"]: obj["position"] for obj in scene["objects"]}


def _move_position(plan, target_object):
    move = next(
        command
        for command in plan.commands
        if command.command_type == CommandType.MOVE
        and command.target_object == target_object
    )
    return [move.target_position.x, move.target_position.y]


def test_current_scene_contains_planner_compatible_objects(monkeypatch):
    # Read the detector payload so get_current_scene() performs the live conversion.
    monkeypatch.setenv("VISION_SCENE_FILE", str(VISION_OUTPUT))

    scene = get_current_scene()

    assert _object_positions(scene) == EXPECTED_POSITIONS
    for obj in scene["objects"]:
        assert set(obj) == {"label", "position"}
        assert isinstance(obj["label"], str)
        assert len(obj["position"]) == 2
        assert all(isinstance(value, (int, float)) for value in obj["position"])


@pytest.mark.parametrize(
    ("instruction", "action", "object_target", "destination"),
    [
        ("pick up the red block", ActionType.PICK, "red block", None),
        ("locate the green block", ActionType.LOCATE, "green block", None),
        ("move the green block to the blue tray", ActionType.MOVE, "green block", "blue tray"),
        ("place the red block in the blue tray", ActionType.PLACE, "red block", "blue tray"),
        (
            "pick up the green block and place it in the blue tray",
            ActionType.PICK,
            "green block",
            "blue tray",
        ),
    ],
)
def test_pipeline_passes_live_vision_labels_and_positions_to_planner(
    monkeypatch,
    tmp_path,
    instruction,
    action,
    object_target,
    destination,
):
    monkeypatch.setenv("VISION_SCENE_FILE", str(VISION_OUTPUT))

    # Parsing is stubbed to keep this vision integration test local and repeatable.
    def parse_sample(raw_instruction):
        assert raw_instruction == instruction
        return ParsedInstruction(
            action=action,
            object_target=object_target,
            destination=destination,
            confidence=ConfidenceLevel.HIGH,
            raw_instruction=raw_instruction,
        )

    planner_scenes = []
    generate_plan = TaskPlanner.generate_plan

    def capture_planner_scene(self, parsed, scene, task_id=None):
        planner_scenes.append(scene)
        return generate_plan(self, parsed, scene, task_id=task_id)

    monkeypatch.setattr("main.parse_instruction", parse_sample)
    monkeypatch.setattr(TaskPlanner, "generate_plan", capture_planner_scene)

    tracker = PipelineTracker(log_path=str(tmp_path / "task_log.json"))
    result = run_pipeline(instruction, verbose=False, tracker=tracker)

    assert result["success"] is True
    assert len(planner_scenes) == 1
    assert _object_positions(planner_scenes[0]) == EXPECTED_POSITIONS
    assert result["plan"].commands[0].target_object == object_target

    if action != ActionType.LOCATE:
        assert _move_position(result["plan"], object_target) == EXPECTED_POSITIONS[object_target]
    if destination:
        assert _move_position(result["plan"], destination) == EXPECTED_POSITIONS[destination]
