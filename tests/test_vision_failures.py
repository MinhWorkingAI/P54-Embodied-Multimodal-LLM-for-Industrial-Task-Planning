import json

from llm_backend.schema import ActionType, ConfidenceLevel, ParsedInstruction
from llm_backend.tracker import PipelineTracker
from main import run_pipeline


def _parsed_instruction(raw_instruction: str) -> ParsedInstruction:
    return ParsedInstruction(
        action=ActionType.PICK,
        object_target="red block",
        destination=None,
        confidence=ConfidenceLevel.HIGH,
        raw_instruction=raw_instruction,
    )


def _logged_task(log_path):
    with open(log_path, encoding="utf-8") as f:
        tasks = json.load(f)
    return tasks[-1]


def test_pipeline_logs_and_exits_when_scene_file_is_missing(monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "task_log.json"
    tracker = PipelineTracker(log_path=str(log_path))

    monkeypatch.setattr("main.parse_instruction", _parsed_instruction)

    def missing_scene():
        raise FileNotFoundError("missing-scene.json")

    monkeypatch.setattr("main.get_current_scene", missing_scene)

    result = run_pipeline("pick up the red block", tracker=tracker)
    output = capsys.readouterr().out
    task = _logged_task(log_path)

    assert result["success"] is False
    assert "Scene file missing" in output
    assert task["status"] == "failed"
    assert task["stages"]["vision_lookup"]["status"] == "failed"
    assert "Scene file missing" in task["stages"]["vision_lookup"]["error"]


def test_pipeline_logs_and_exits_when_scene_has_no_objects(monkeypatch, tmp_path, capsys):
    log_path = tmp_path / "task_log.json"
    tracker = PipelineTracker(log_path=str(log_path))

    monkeypatch.setattr("main.parse_instruction", _parsed_instruction)
    monkeypatch.setattr("main.get_current_scene", lambda: {"objects": []})

    result = run_pipeline("pick up the red block", tracker=tracker)
    output = capsys.readouterr().out
    task = _logged_task(log_path)

    assert result["success"] is False
    assert "No objects detected" in output
    assert task["status"] == "failed"
    assert task["stages"]["vision_lookup"]["status"] == "failed"
    assert task["stages"]["vision_lookup"]["payload"] == {
        "object_count": 0,
        "objects": [],
    }
