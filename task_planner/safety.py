"""
safety.py
---------
Bounds validation helpers for the task planner.
"""

from simulation_backend.action_schema import Position


class BoundaryViolation(ValueError):
    """Raised when a target position is outside the allowed workspace bounds."""


def validate_bounds(position: Position) -> None:
    """
    Validate a position against the robot workspace bounds.

    Bounds:
        X in [0.1, 0.8]
        Y in [-0.4, 0.4]
        Z in [0.0, 0.5]

    Raises:
        BoundaryViolation: If any coordinate is outside the allowed range.
    """
    x, y, z = position.x, position.y, position.z
    if not (0.1 <= x <= 0.8):
        raise BoundaryViolation(f"X out of bounds: {x}")
    if not (-0.4 <= y <= 0.4):
        raise BoundaryViolation(f"Y out of bounds: {y}")
    if not (0.0 <= z <= 0.5):
        raise BoundaryViolation(f"Z out of bounds: {z}")
