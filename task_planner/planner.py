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
import sys
import os
from llm_backend.schema import ParsedInstruction, ActionType
from execution.action_schema import ActionPlan, RobotCommand, CommandType, Position

logger = logging.getLogger(__name__)


# ── Scene helper ───────────────────────────────────────────────────────────────
def _find_in_scene(scene: dict, query: str) -> dict | None:
    """
    Find an object in the scene by label (case-insensitive partial match).

    Scene format expected:
    {
        "objects": [
            {"label": "red block",  "position": (2.5, 1.0)},
            {"label": "left tray",  "position": (4.0, 0.5)},
        ]
    }
    """
    query_lower = query.lower()
    for obj in scene.get("objects", []):
        label = (obj.get("label") or obj.get("name", "")).lower()
        if query_lower in label or label in query_lower:
            pos = obj.get("position", (0.0, 0.0))
            if isinstance(pos, (list, tuple)):
                position = (float(pos[0]), float(pos[1]))
            elif isinstance(pos, dict):
                position = (float(pos.get("x", 0)), float(pos.get("y", 0)))
            else:
                position = (0.0, 0.0)
            return {"label": obj.get("label") or obj.get("name"), "position": position}

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
    ) -> ActionPlan:
        """
        Generate a step-by-step ActionPlan from instruction and scene.

        Args:
            parsed:  ParsedInstruction from llm_module.parser
            scene:   Scene dict from vision module
            task_id: Optional tracker task_id for cross-domain logging

        Returns:
            ActionPlan ready to be sent to the executor

        Raises:
            ValueError: If required objects cannot be found in the scene
        """
        action = parsed.action.value

        logger.info(f"Planning: action={action}, object={parsed.object_target}, "
                    f"destination={parsed.destination}")

        if action == ActionType.PICK.value:
            steps = self._plan_pick(parsed, scene)

        elif action == ActionType.PLACE.value:
            steps = self._plan_place(parsed, scene)

        elif action == ActionType.MOVE.value:
            steps = self._plan_move(parsed, scene)

        elif action == ActionType.LOCATE.value:
            steps = self._plan_locate(parsed, scene)

        else:
            raise ValueError(f"Unknown action type: {action}")

        return ActionPlan(
            task_id=task_id,
            instruction=parsed.raw_instruction,
            commands=steps,
        )

    # ── Planning methods ───────────────────────────────────────────────────────

    def _plan_pick(self, parsed: ParsedInstruction, scene: dict) -> list[RobotCommand]:
        """
        Pick plan:  locate → move to object → pick
        If destination is also specified: → move to destination → place
        """
        steps   = []
        step_n  = 1
        obj_name = parsed.object_target

        # Find object in scene
        obj = _find_in_scene(scene, obj_name)
        if obj is None:
            raise ValueError(
                f"Object '{obj_name}' not found in scene. "
                f"Available: {[o.get('label') for o in scene.get('objects', [])]}"
            )

        obj_pos = Position(x=obj["position"][0], y=obj["position"][1])

        # Step 1: Locate
        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.LOCATE,
            target_object=obj_name,
            description=f"Confirm '{obj_name}' is in scene at {obj['position']}",
        ))
        step_n += 1

        # Step 2: Move to object
        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.MOVE,
            target_object=obj_name,
            target_position=obj_pos,
            description=f"Navigate arm to '{obj_name}'",
        ))
        step_n += 1

        # Step 3: Pick
        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.PICK,
            target_object=obj_name,
            description=f"Grasp '{obj_name}'",
        ))
        step_n += 1

        # Optional: if destination is specified, complete the place too
        if parsed.destination:
            dest_name = parsed.destination
            dest      = _find_in_scene(scene, dest_name)
            if dest:
                dest_pos = Position(x=dest["position"][0], y=dest["position"][1])
                steps.append(RobotCommand(
                    step=step_n, command_type=CommandType.MOVE,
                    target_object=dest_name,
                    target_position=dest_pos,
                    description=f"Navigate to '{dest_name}'",
                ))
                step_n += 1
                steps.append(RobotCommand(
                    step=step_n, command_type=CommandType.PLACE,
                    target_object=dest_name,
                    description=f"Place '{obj_name}' at '{dest_name}'",
                ))

        return steps

    def _plan_place(self, parsed: ParsedInstruction, scene: dict) -> list[RobotCommand]:
        """
        Place plan:  locate object → move to object → pick → move to dest → place
        """
        steps    = []
        step_n   = 1
        obj_name  = parsed.object_target
        dest_name = parsed.destination or "right tray"

        obj  = _find_in_scene(scene, obj_name)
        dest = _find_in_scene(scene, dest_name)

        if obj is None:
            raise ValueError(f"Object '{obj_name}' not found in scene")
        if dest is None:
            raise ValueError(f"Destination '{dest_name}' not found in scene")

        obj_pos  = Position(x=obj["position"][0],  y=obj["position"][1])
        dest_pos = Position(x=dest["position"][0], y=dest["position"][1])

        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.LOCATE,
            target_object=obj_name,
            description=f"Confirm '{obj_name}' exists at {obj['position']}",
        ))
        step_n += 1

        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.MOVE,
            target_object=obj_name, target_position=obj_pos,
            description=f"Navigate to '{obj_name}'",
        ))
        step_n += 1

        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.PICK,
            target_object=obj_name,
            description=f"Grasp '{obj_name}'",
        ))
        step_n += 1

        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.MOVE,
            target_object=dest_name, target_position=dest_pos,
            description=f"Navigate to '{dest_name}'",
        ))
        step_n += 1

        steps.append(RobotCommand(
            step=step_n, command_type=CommandType.PLACE,
            target_object=dest_name,
            description=f"Release '{obj_name}' at '{dest_name}'",
        ))

        return steps

    def _plan_move(self, parsed: ParsedInstruction, scene: dict) -> list[RobotCommand]:
        """
        Move plan:  locate object → pick → move to destination → place
        """
        # Move is functionally the same as place for this robot
        return self._plan_place(parsed, scene)

    def _plan_locate(self, parsed: ParsedInstruction, scene: dict) -> list[RobotCommand]:
        """
        Locate plan:  just locate the object and report position
        """
        obj_name = parsed.object_target
        obj      = _find_in_scene(scene, obj_name)

        if obj is None:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        return [RobotCommand(
            step=1, command_type=CommandType.LOCATE,
            target_object=obj_name,
            description=f"Find '{obj_name}' at {obj['position']}",
        )]