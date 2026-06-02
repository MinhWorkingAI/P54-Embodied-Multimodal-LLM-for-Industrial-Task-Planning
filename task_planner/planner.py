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

# Home / neutral pose the arm returns to at the end of every task (RESET step).
HOME_POSITION = Position(x=0.0, y=0.0, z=0.0)

# Spatial-relation offsets (in workspace units). The planner drops the carried
# object *relative to* a reference object, e.g. "left of the blue block".
SPATIAL_OFFSET = 0.15   # lateral shift for left / right / front / behind
NEAR_OFFSET = 0.10      # smaller shift for "near" / "next to"
STACK_HEIGHT = 0.10     # vertical shift for "on top of" / "above"


# ── Spatial-relation resolver ──────────────────────────────────────────────────
def _apply_spatial_offset(base: Position, relation: str | None) -> Position:
    """
    Translate a destination position according to a parsed spatial relation.

    This is what makes the planner *spatially aware*: given a reference object's
    coordinates and a relation phrase from the LLM, it returns the adjusted drop
    point. For example, "left of the blue block" shifts the target -X of the blue
    block's centre so the object is placed to its left.

    Recognised relations (case- and separator-insensitive, so "left of",
    "left_of" and "LEFT OF" are equivalent):

        left of / left          →  -X by SPATIAL_OFFSET
        right of / right        →  +X by SPATIAL_OFFSET
        in front of / front     →  -Y by SPATIAL_OFFSET
        behind / back           →  +Y by SPATIAL_OFFSET
        near / next to / beside →  +X by NEAR_OFFSET
        on top of / above / on  →  +Z by STACK_HEIGHT (stack onto reference)
        inside / in / into      →  no offset (drop into the container centre)

    A missing or unrecognised relation returns ``base`` unchanged, so callers
    can pass ``parsed.spatial_relation`` directly without pre-checking it.
    """
    if not relation:
        return base

    # Normalise: lowercase, collapse underscores to spaces, strip padding.
    key = relation.strip().lower().replace("_", " ")

    if key in ("left of", "left", "to the left of"):
        return Position(x=base.x - SPATIAL_OFFSET, y=base.y, z=base.z)
    if key in ("right of", "right", "to the right of"):
        return Position(x=base.x + SPATIAL_OFFSET, y=base.y, z=base.z)
    if key in ("in front of", "front", "ahead of"):
        return Position(x=base.x, y=base.y - SPATIAL_OFFSET, z=base.z)
    if key in ("behind", "back", "back of"):
        return Position(x=base.x, y=base.y + SPATIAL_OFFSET, z=base.z)
    if key in ("near", "next to", "beside", "close to"):
        return Position(x=base.x + NEAR_OFFSET, y=base.y, z=base.z)
    if key in ("on top of", "on", "above", "over"):
        return Position(x=base.x, y=base.y, z=base.z + STACK_HEIGHT)
    if key in ("inside", "in", "into", "within"):
        return Position(x=base.x, y=base.y, z=base.z)

    # Unknown relation: leave the destination untouched.
    return base


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

        # The primitive plan is already the canonical, ordered sequence
        # (Locate → … → Reset for pick-and-place tasks), so we simply map each
        # primitive to its executor-level command one-to-one. The leading LOCATE
        # and trailing RESET are produced by the decomposition itself.
        commands: list[RobotCommand] = []
        for step_n, prim in enumerate(primitive_plan.steps, start=1):
            cmd_type = self._map_primitive_to_command(prim.action)
            commands.append(RobotCommand(
                step=step_n,
                command_type=cmd_type,
                target_object=prim.target_object,
                target_position=prim.target_position,
                parameters=prim.parameters,
                description=prim.description,
            ))

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
            # Apply the spatial relation ("left of", "on top of", …) so the drop
            # point is computed relative to the reference object's coordinates.
            dest_pos = _apply_spatial_offset(dest["position"], parsed.spatial_relation)

        steps = self._decompose_pick_and_place(parsed.object_target, obj_pos, dest_name, dest_pos)

        if enforce_bounds:
            for step in steps:
                # The RESET step targets the home pose, which lies outside the
                # task workspace by design, so it is exempt from bounds checks.
                if step.target_position is not None and step.action != PrimitiveActionType.RESET:
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
        """
        Decompose a (multi-step) manipulation task into ordered primitives.

        Full pick-and-place (a destination is supplied) yields the canonical
        9-step sequence used throughout the project:

            1. LOCATE   confirm the target object exists in the scene
            2. APPROACH move above the object at a safe height
            3. LOWER    descend to grasp height
            4. GRASP    close the gripper on the object
            5. LIFT     raise the object clear of the surface
            6. MOVE     travel to the (spatially-resolved) destination
            7. LOWER    descend to placement height at the destination
            8. RELEASE  open the gripper to drop the object
            9. RESET    return the arm to its home/neutral pose

        A pick-only task (no destination) skips steps 6–8 and ends with RESET.
        """
        steps: list[PrimitiveAction] = []

        # 1. LOCATE — verify the object is present before any motion.
        steps.append(PrimitiveAction(
            action=PrimitiveActionType.LOCATE,
            target_object=obj_name,
            target_position=Position(x=obj_pos.x, y=obj_pos.y, z=obj_pos.z),
            description=f"Locate '{obj_name}'",
        ))
        # 2. APPROACH — hover above the object.
        steps.append(PrimitiveAction(
            action=PrimitiveActionType.APPROACH,
            target_object=obj_name,
            target_position=Position(x=obj_pos.x, y=obj_pos.y, z=APPROACH_Z),
            description=f"Approach '{obj_name}'",
        ))
        # 3. LOWER — descend to grasp height.
        steps.append(PrimitiveAction(
            action=PrimitiveActionType.LOWER,
            target_object=obj_name,
            target_position=Position(x=obj_pos.x, y=obj_pos.y, z=GRASP_Z),
            description=f"Lower to '{obj_name}'",
        ))
        # 4. GRASP — close the gripper.
        steps.append(PrimitiveAction(
            action=PrimitiveActionType.GRASP,
            target_object=obj_name,
            description=f"Grasp '{obj_name}'",
        ))
        # 5. LIFT — raise the object clear of the surface.
        steps.append(PrimitiveAction(
            action=PrimitiveActionType.LIFT,
            target_object=obj_name,
            target_position=Position(x=obj_pos.x, y=obj_pos.y, z=LIFT_Z),
            description=f"Lift '{obj_name}'",
        ))

        if dest_pos is not None:
            # Placement height: honour an explicit z from the spatial relation
            # (e.g. "on top of" raises z), otherwise use the standard grasp height.
            place_z = dest_pos.z if dest_pos.z > 0 else GRASP_Z
            # 6. MOVE — travel above the resolved destination.
            steps.append(PrimitiveAction(
                action=PrimitiveActionType.MOVE,
                target_object=dest_name,
                target_position=Position(x=dest_pos.x, y=dest_pos.y, z=LIFT_Z),
                description=f"Move to '{dest_name}'",
            ))
            # 7. LOWER — descend to placement height at the destination.
            steps.append(PrimitiveAction(
                action=PrimitiveActionType.LOWER,
                target_object=dest_name,
                target_position=Position(x=dest_pos.x, y=dest_pos.y, z=place_z),
                description=f"Lower at '{dest_name}'",
            ))
            # 8. RELEASE — open the gripper to place the object.
            steps.append(PrimitiveAction(
                action=PrimitiveActionType.RELEASE,
                target_object=dest_name,
                description=f"Release at '{dest_name}'",
            ))

        # 9. RESET — return the arm to its home pose (always the final step).
        steps.append(PrimitiveAction(
            action=PrimitiveActionType.RESET,
            target_position=HOME_POSITION,
            description="Reset arm to home pose",
        ))

        return steps

    @staticmethod
    def _map_primitive_to_command(action: PrimitiveActionType) -> CommandType:
        # RESET is a Cartesian move back to the home pose, so it maps to MOVE.
        if action in (PrimitiveActionType.APPROACH, PrimitiveActionType.LOWER,
                      PrimitiveActionType.LIFT, PrimitiveActionType.MOVE,
                      PrimitiveActionType.RESET):
            return CommandType.MOVE
        if action == PrimitiveActionType.GRASP:
            return CommandType.PICK
        if action == PrimitiveActionType.RELEASE:
            return CommandType.PLACE
        return CommandType.LOCATE
