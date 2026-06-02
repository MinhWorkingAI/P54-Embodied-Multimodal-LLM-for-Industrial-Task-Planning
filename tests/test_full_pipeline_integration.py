import json
from pathlib import Path

import pytest

from llm_backend.schema import ActionType, ConfidenceLevel, ParsedInstruction
from llm_backend.tracker import PIPELINE_STAGES, PipelineTracker
from main import run_pipeline
from vision_backend.scene_representation import get_current_scene


VISION_OUTPUT = Path(__file__).resolve().parents[1] / "drafts" / "scene_output.json"


@pytest.fixture
def live_scene(monkeypatch):
    # Use the detector output so the real vision adapter builds the planner scene.
    monkeypatch.setenv("VISION_SCENE_FILE", str(VISION_OUTPUT))
    return get_current_scene()


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
def test_instruction_runs_through_all_pipeline_stages(
    monkeypatch,
    tmp_path,
    live_scene,
    instruction,
    action,
    object_target,
    destination,
):
    assert live_scene["objects"]

    # Stub only the external LLM call. Vision, planning, execution, and feedback stay real.
    monkeypatch.setattr(
        "main.parse_instruction",
        lambda raw_instruction: ParsedInstruction(
            action=action,
            object_target=object_target,
            destination=destination,
            confidence=ConfidenceLevel.HIGH,
            raw_instruction=raw_instruction,
        ),
    )

    log_path = tmp_path / "task_log.json"
    tracker = PipelineTracker(log_path=str(log_path))
    result = run_pipeline(instruction, verbose=False, tracker=tracker)

    assert result["success"] is True

    with log_path.open(encoding="utf-8") as log_file:
        logged_tasks = json.load(log_file)

    assert len(logged_tasks) == 1
    logged_task = logged_tasks[0]
    assert logged_task["instruction"] == instruction
    assert logged_task["status"] == "completed"
    assert set(logged_task["stages"]) == set(PIPELINE_STAGES)
    assert all(
        logged_task["stages"][stage]["status"] == "success"
        for stage in PIPELINE_STAGES
    )
