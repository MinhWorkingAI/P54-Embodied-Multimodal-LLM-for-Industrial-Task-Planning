"""
vision_backend/scene_representation.py
<<<<<<< HEAD
--------------------------------------
Production scene adapter for the main pipeline.

The public contract is intentionally small:

    get_current_scene() -> {"objects": [{"label": str, "position": [x, y, z]}]}

The data comes from the live simulation vision stack.  The Simulation class owns
camera capture, YOLO/colour detector selection, ground-truth fallback, and
conversion through SceneBuilder.
"""

from __future__ import annotations

import os
from typing import Any


def _normalise_position(position: Any) -> list[float]:
    """Return a JSON-compatible [x, y, z] position list."""
    if isinstance(position, dict):
        return [
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
        ]

    if isinstance(position, (list, tuple)):
        if len(position) < 2:
            return [0.0, 0.0, 0.0]
        z = position[2] if len(position) > 2 else 0.0
        return [float(position[0]), float(position[1]), float(z)]

    return [0.0, 0.0, 0.0]


def _normalise_planner_scene(scene: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Keep only the planner-facing scene contract and make it literal JSON data.
    """
    objects = []
    for obj in scene.get("objects", []):
        label = obj.get("label") or obj.get("name")
        if not label:
            continue
        objects.append({
            "label": str(label),
            "position": _normalise_position(obj.get("position", [0.0, 0.0, 0.0])),
        })
    return {"objects": objects}


def _validate_all_workspace_objects(scene: dict[str, Any], sim: Any) -> None:
    """Fail fast if the live scene omits an object registered in the workspace."""
    registry = getattr(sim, "registry", None)
    if registry is None:
        return

    expected = [entry.label for entry in registry.all_entries()]
    found = {obj["label"] for obj in scene.get("objects", [])}
    missing = [label for label in expected if label not in found]

    if missing:
        raise RuntimeError(
            "Live vision scene is missing workspace object(s): "
            + ", ".join(missing)
        )


def get_current_scene(verbose: bool = False, sim: Any | None = None) -> dict[str, Any]:
    """
    Capture the current workspace through the real simulation vision pipeline.

    Args:
        verbose: Print the detector summary emitted by Simulation.get_live_scene().
        sim: Optional existing Simulation instance. If omitted, a temporary
             headless Simulation is created and disconnected before returning.

    Returns:
        Planner-compatible, JSON-serialisable scene dict:
        {"objects": [{"label": "red block", "position": [x, y, z]}, ...]}
    """
    os.environ.setdefault("VISION_DETECTOR", "yolo")

    owns_sim = sim is None
    if sim is None:
        from simulation_backend.simulation import Simulation
        sim = Simulation()

    try:
        scene = _normalise_planner_scene(sim.get_live_scene(verbose=verbose))
        _validate_all_workspace_objects(scene, sim)
        return scene
    finally:
        if owns_sim:
            sim.disconnect()


def get_planner_scene(verbose: bool = False) -> dict[str, Any]:
    """Backward-compatible alias used by older imports."""
    return get_current_scene(verbose=verbose)


def build_scene_representation(vision_output: dict[str, Any]) -> dict[str, Any]:
    """
    Convert older vision-output dictionaries into the planner scene format.

    This keeps historical tests/imports working while production code uses
    get_current_scene().
    """
    if "objects" in vision_output:
        return _normalise_planner_scene(vision_output)

    objects = []
    for obj in vision_output.get("detected_objects", []):
        label = obj.get("label") or obj.get("name")
        if not label:
            continue
        position = obj.get("position", {})
        coords = position.get("coordinates_3d", position) if isinstance(position, dict) else position
        objects.append({"label": str(label), "position": _normalise_position(coords)})
    return {"objects": objects}


def calculate_center(bbox: dict[str, float] | list[float] | tuple[float, ...]) -> tuple[float, float]:
    """Return the centre point of a 2D bounding box."""
    if isinstance(bbox, dict):
        x1 = bbox.get("x_min", bbox.get("xmin", bbox.get("x1", 0.0)))
        y1 = bbox.get("y_min", bbox.get("ymin", bbox.get("y1", 0.0)))
        x2 = bbox.get("x_max", bbox.get("xmax", bbox.get("x2", 0.0)))
        y2 = bbox.get("y_max", bbox.get("ymax", bbox.get("y2", 0.0)))
    else:
        x1, y1, x2, y2 = bbox[:4]
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
=======
---------------------------------------
JSON-based scene provider for the pipeline's Vision Lookup stage.

Reads a static scene JSON file from disk and converts it into the
planner-compatible scene dict that main.py and task_planner/planner.py expect.

Public interface:
    get_current_scene() -> dict

Input:
    JSON file at vision_backend/scene_representation.json (default)
    or path set via VISION_SCENE_FILE environment variable.

    Expected JSON format:
        {
            "objects": [
                {"label": "red block", "position": [150.0, 110.0]},
                ...
            ]
        }

Output:
    {
        "objects": [
            {"label": "red block", "position": [150.0, 110.0]},
            ...
        ]
    }

Usage:
    from vision_backend.scene_representation import get_current_scene
    scene = get_current_scene()
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_SCENE_FILE = Path(__file__).resolve().parent / "scene_representation.json"


def get_current_scene() -> dict:
    """
    Read the static scene JSON file and return a planner-compatible scene dict.

    Returns:
        {"objects": [{"label": str, "position": [x, y]}, ...]}

    Raises:
        FileNotFoundError : Scene file does not exist.
        ValueError        : File is not valid JSON or missing "objects" key.
    """
    scene_file = Path(os.getenv("VISION_SCENE_FILE", str(_DEFAULT_SCENE_FILE)))

    if not scene_file.exists():
        raise FileNotFoundError(
            f"Scene file not found: {scene_file}. "
            f"Set VISION_SCENE_FILE in .env or ensure vision_backend/scene_representation.json exists."
        )

    try:
        with open(scene_file, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Scene file is not valid JSON: {scene_file} — {e}") from e

    if "objects" not in raw:
        raise ValueError(f"Scene file missing required 'objects' key: {scene_file}")

    scene = _convert(raw)
    logger.info(f"[scene_representation] Loaded {len(scene['objects'])} objects from {scene_file}")
    return scene


def _convert(raw: dict) -> dict:
    """
    Normalise raw JSON into planner-compatible scene format.
    Strips Z if position has 3 elements.
    """
    objects = []
    for obj in raw["objects"]:
        label = obj["label"]
        position = list(obj["position"])[:2]
        objects.append({"label": label, "position": position})
    return {"objects": objects}
>>>>>>> ff7e0ebec2ae77f42897a7eaf9886df39b387f04
