"""
task_planner/planner.py
-----------------------
PB7: Task planning module.

Combines ParsedInstruction (from LLM module) and a scene representation
(from vision module) to generate a step-by-step robot action plan.

Uses rule-based planning for Sprint 2 — deterministic, testable without
API calls, and sufficient for pick-and-place with coloured blocks.
An LLM-based planner can replace this in Sprint 3 if needed.

Standard action sequence:
    locate   → confirm object exists in scene
    move     → navigate arm to object position
    pick     → grasp the object
    move     → navigate to destination
    place    → release object at destination

Usage:
    from task_planner.planner import TaskPlanner

    planner = TaskPlanner()
    plan    = planner.generate_plan(parsed_instruction, scene)
    plan.print_plan()
"""

import logging
from llm_backend.schema import ParsedInstruction, ActionType
from simulation_backend.action_schema import ActionPlan, RobotCommand, CommandType, Position
from task_planner.schema import (
    PlannerInput,
    PlannerOutput,
    PrimitiveAction,
    PrimitiveActionType,
)
from task_planner.safety import validate_bounds

logger = logging.getLogger(__name__)


# ── Planner constants ─────────────────────────────────────────────────────────
APPROACH_Z = 0.30
GRASP_Z = 0.05
LIFT_Z = 0.30


# ── Scene helpers ─────────────────────────────────────────────────────────────
def _extract_position(raw: object) -> Position:
    if isinstance(raw, Position):
        return raw
    if isinstance(raw, dict):
        if "coordinates_3d" in raw:
            raw = raw["coordinates_3d"]
        if "center" in raw:
            raw = raw["center"]
        if isinstance(raw, dict):
            return Position(
                x=float(raw.get("x", 0.0)),
                y=float(raw.get("y", 0.0)),
                z=float(raw.get("z", 0.0)),
            )
    if isinstance(raw, (list, tuple)):
        x = float(raw[0]) if len(raw) > 0 else 0.0
        y = float(raw[1]) if len(raw) > 1 else 0.0
        z = float(raw[2]) if len(raw) > 2 else 0.0
        return Position(x=x, y=y, z=z)
    return Position(x=0.0, y=0.0, z=0.0)


def _iter_scene_objects(scene: dict) -> list[tuple[str, object]]:
    if "objects" in scene:
        items = []
        for obj in scene.get("objects", []):
            label = obj.get("label") or obj.get("name") or ""
            position = obj.get("position") or obj.get("center") or obj
            items.append((label, position))
        return items

    if "scene" in scene and isinstance(scene["scene"], dict):
        items = []
        for label, data in scene["scene"].items():
            items.append((label, data))
        return items

    if "detected_objects" in scene:
        items = []
        for obj in scene.get("detected_objects", []):
            label = obj.get("label") or obj.get("name") or ""
            position = obj.get("position") or obj
            items.append((label, position))
        return items

    return []


def _find_in_scene(scene: dict, query: str) -> dict | None:
    """
    Find an object in the scene by label (case-insensitive partial match).

    Supported scene formats:
      - {"objects": [{"label": "red block", "position": (2.5, 1.0)}]}
      - {"scene": {"red block": {"center": [150, 110]}}}
      - {"detected_objects": [{"label": "block", "position": {"coordinates_3d": {...}}}]}
    """
    query_lower = query.lower()
    for label, raw_position in _iter_scene_objects(scene):
        label_lower = (label or "").lower()
        if query_lower in label_lower or label_lower in query_lower:
            position = _extract_position(raw_position)
            return {"label": label, "position": position}

    return None


# ── Task planner ───────────────────────────────────────────────────────────────
class TaskPlanner:
    """
    Rule-based task planner for Sprint 2.

    Converts a ParsedInstruction + scene into an ordered ActionPlan.
    Each action type has its own planning method so it is easy to
    extend or override individual behaviours.
    """

    def generate_plan(
        self,
        parsed: ParsedInstruction,
        scene: dict,
        task_id: str | None = None,
        enforce_bounds: bool = False,
    ) -> ActionPlan:
        """
        Generate a step-by-step ActionPlan from instruction and scene.

        Args:
            parsed:          ParsedInstruction from llm_module.parser
            scene:           Scene dict from vision module
            task_id:         Optional tracker task_id for cross-domain logging
            enforce_bounds:  When True, validate target positions against workspace bounds

        Returns:
            ActionPlan ready to be sent to the executor

        Raises:
            ValueError: If required objects cannot be found in the scene
        """
        planner_input = PlannerInput(parsed_instruction=parsed, scene=scene)
        primitive_plan = self.generate_primitive_plan(
            planner_input,
            task_id=task_id,
            enforce_bounds=enforce_bounds,
        )

        commands: list[RobotCommand] = []
        step_n = 1

        if parsed.action != ActionType.LOCATE:
            commands.append(RobotCommand(
                step=step_n,
                command_type=CommandType.LOCATE,
                target_object=parsed.object_target,
                description=f"Confirm '{parsed.object_target}' is in scene",
            ))
            step_n += 1

        for prim in primitive_plan.steps:
            cmd_type = self._map_primitive_to_command(prim.action)
            commands.append(RobotCommand(
                step=step_n,
                command_type=cmd_type,
                target_object=prim.target_object,
                target_position=prim.target_position,
                parameters=prim.parameters,
                description=prim.description,
            ))
            step_n += 1

        return ActionPlan(
            task_id=task_id,
            instruction=parsed.raw_instruction,
            commands=commands,
        )

    def generate_primitive_plan(
        self,
        planner_input: PlannerInput,
        task_id: str | None = None,
        enforce_bounds: bool = False,
    ) -> PlannerOutput:
        """
        Generate a primitive action plan from a PlannerInput schema.
        """
        parsed = planner_input.parsed_instruction
        scene = planner_input.scene

        logger.info(
            f"Planning primitives: action={parsed.action.value}, "
            f"object={parsed.object_target}, destination={parsed.destination}"
        )

        if parsed.action == ActionType.LOCATE:
            obj = _find_in_scene(scene, parsed.object_target)
            if obj is None:
                raise ValueError(
                    f"Object '{parsed.object_target}' not found in scene. "
                    f"Available: {[label for label, _ in _iter_scene_objects(scene)]}"
                )
            pos = obj["position"]
            if enforce_bounds:
                validate_bounds(pos)
            steps = [PrimitiveAction(
                action=PrimitiveActionType.LOCATE,
                target_object=parsed.object_target,
                target_position=pos,
                description=f"Locate '{parsed.object_target}'",
            )]
            return PlannerOutput(
                instruction=parsed.raw_instruction,
                steps=steps,
                task_id=task_id,
            )

        obj = _find_in_scene(scene, parsed.object_target)
        if obj is None:
            raise ValueError(
                f"Object '{parsed.object_target}' not found in scene. "
                f"Available: {[label for label, _ in _iter_scene_objects(scene)]}"
            )
        obj_pos = obj["position"]

        dest_name = parsed.destination
        if parsed.action in (ActionType.PLACE, ActionType.MOVE) and not dest_name:
            dest_name = "right tray"

        dest_pos = None
        if dest_name:
            dest = _find_in_scene(scene, dest_name)
            if dest is None:
                raise ValueError(f"Destination '{dest_name}' not found in scene")
            dest_pos = dest["position"]

        steps = self._decompose_pick_and_place(parsed.object_target, obj_pos, dest_name, dest_pos)

        if enforce_bounds:
            for step in steps:
                if step.target_position is not None:
                    validate_bounds(step.target_position)

        return PlannerOutput(
            instruction=parsed.raw_instruction,
            steps=steps,
            task_id=task_id,
        )

    def _decompose_pick_and_place(
        self,
        obj_name: str,
        obj_pos: Position,
        dest_name: str | None,
        dest_pos: Position | None,
    ) -> list[PrimitiveAction]:
        steps: list[PrimitiveAction] = []

        steps.append(PrimitiveAction(
            action=PrimitiveActionType.APPROACH,
            target_object=obj_name,
            target_position=Position(x=obj_pos.x, y=obj_pos.y, z=APPROACH_Z),
            description=f"Approach '{obj_name}'",
        ))
        steps.append(PrimitiveAction(
            action=PrimitiveActionType.LOWER,
            target_object=obj_name,
            target_position=Position(x=obj_pos.x, y=obj_pos.y, z=GRASP_Z),
            description=f"Lower to '{obj_name}'",
        ))
        steps.append(PrimitiveAction(
            action=PrimitiveActionType.GRASP,
            target_object=obj_name,
            description=f"Grasp '{obj_name}'",
        ))
        steps.append(PrimitiveAction(
            action=PrimitiveActionType.LIFT,
            target_object=obj_name,
            target_position=Position(x=obj_pos.x, y=obj_pos.y, z=LIFT_Z),
            description=f"Lift '{obj_name}'",
        ))

        if dest_pos is not None:
            steps.append(PrimitiveAction(
                action=PrimitiveActionType.MOVE,
                target_object=dest_name,
                target_position=Position(x=dest_pos.x, y=dest_pos.y, z=LIFT_Z),
                description=f"Move to '{dest_name}'",
            ))
            steps.append(PrimitiveAction(
                action=PrimitiveActionType.RELEASE,
                target_object=dest_name,
                description=f"Release at '{dest_name}'",
            ))

        return steps

    @staticmethod
    def _map_primitive_to_command(action: PrimitiveActionType) -> CommandType:
        if action in (PrimitiveActionType.APPROACH, PrimitiveActionType.LOWER,
                      PrimitiveActionType.LIFT, PrimitiveActionType.MOVE):
            return CommandType.MOVE
        if action == PrimitiveActionType.GRASP:
            return CommandType.PICK
        if action == PrimitiveActionType.RELEASE:
            return CommandType.PLACE
        return CommandType.LOCATE
