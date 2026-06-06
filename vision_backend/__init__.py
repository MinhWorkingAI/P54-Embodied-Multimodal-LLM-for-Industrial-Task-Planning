"""
vision_backend/
---------------
Vision module for the Multimodal LLM Industrial Task Planning system.
Sprint 1 — PB3 / PB4 / PB5

Public interface:
    from vision_backend.scene_representation  import build_scene_representation, get_planner_scene
    from vision_backend.spatial_relationships import compute_relationships, get_spatial_relationships
    from vision_backend.vision_output         import create_detection, create_scene_output
"""

from vision_backend.scene_representation  import (
    build_scene_representation,
    get_planner_scene,
    calculate_center,
)
from vision_backend.spatial_relationships import (
    compute_relationships,
    get_spatial_relationships,
    build_spatial_output,
)
from vision_backend.vision_output import (
    create_detection,
    create_scene_output,
)

__all__ = [
    "build_scene_representation",
    "get_planner_scene",
    "calculate_center",
    "compute_relationships",
    "get_spatial_relationships",
    "build_spatial_output",
    "create_detection",
    "create_scene_output",
]
