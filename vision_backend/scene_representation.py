from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE_REPRESENTATION = PROJECT_ROOT / "drafts" / "scene_representation.json"
DEFAULT_VISION_OUTPUT = PROJECT_ROOT / "drafts" / "scene_output.json"


def _resolve_scene_file(filename: str | os.PathLike[str]) -> Path:
    path = Path(filename)
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "drafts" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return PROJECT_ROOT / path


def load_json(filename: str | os.PathLike[str]) -> dict[str, Any]:
    with _resolve_scene_file(filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def calculate_center(bounding_box: list[int | float]) -> list[float]:
    x, y, width, height = bounding_box
    return [float(x) + float(width) / 2, float(y) + float(height) / 2]


def build_scene_representation(vision_output: dict[str, Any]) -> dict[str, Any]:
    scene = {}

    for obj in vision_output.get("objects", []):
        label = obj["label"]
        bounding_box = obj["bounding_box"]
        scene[label] = {
            "center": calculate_center(bounding_box),
            "bounding_box": bounding_box,
            "confidence": obj.get("confidence"),
        }

    return {
        "image_id": vision_output.get("image_id", "unknown"),
        "scene": scene,
    }


def _scene_representation_to_planner_scene(scene_data: dict[str, Any]) -> dict[str, Any]:
    planner_objects = []

    for label, details in scene_data.get("scene", {}).items():
        center = details.get("center", [0, 0])
        planner_objects.append({
            "label": label,
            "position": [center[0], center[1]],
        })

    return {"objects": planner_objects}


def get_planner_scene(filename: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    scene_file = filename or os.getenv("VISION_SCENE_FILE")

    if scene_file:
        scene_data = load_json(scene_file)
    elif DEFAULT_SCENE_REPRESENTATION.exists():
        scene_data = load_json(DEFAULT_SCENE_REPRESENTATION)
    elif DEFAULT_VISION_OUTPUT.exists():
        scene_data = build_scene_representation(load_json(DEFAULT_VISION_OUTPUT))
    else:
        raise FileNotFoundError(
            "No vision scene file found. Set VISION_SCENE_FILE or provide "
            "drafts/scene_representation.json."
        )

    if "scene" not in scene_data and "objects" in scene_data:
        scene_data = build_scene_representation(scene_data)

    return _scene_representation_to_planner_scene(scene_data)


def get_current_scene(filename: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """
    Return the latest vision scene in the planner-compatible format.

    Planner format:
    {
        "objects": [
            {"label": "red block", "position": [x, y]}
        ]
    }
    """
    return get_planner_scene(filename)
