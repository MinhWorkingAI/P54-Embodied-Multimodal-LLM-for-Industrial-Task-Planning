"""
test_sprint2.py
---------------
Unit tests for all Sprint 2 modules.
All tests run without API calls or PyBullet.

Run:
    pytest tests/test_sprint2.py -v

Coverage:
    - simulation_backend/action_schema.py  (RobotCommand, ActionPlan, plan_to_commands)
    - simulation_backend/mock_robot.py     (MockRobot all commands, state, edge cases)
    - simulation_backend/executor.py       (Executor success/failure flows)
    - task_planner/planner.py              (TaskPlanner all action types)
"""

import pytest
from unittest.mock import MagicMock

from llm_backend.schema import ParsedInstruction, ActionType, ConfidenceLevel
from simulation_backend.action_schema import (
    RobotCommand, ActionPlan, CommandType, Position, plan_to_commands
)
from simulation_backend.mock_robot import MockRobot, CommandResult
from simulation_backend.executor   import Executor
from task_planner.planner import (
    TaskPlanner,
    SPATIAL_OFFSET, NEAR_OFFSET, STACK_HEIGHT, HOME_POSITION,
)
from task_planner.schema import PlannerInput, PrimitiveActionType


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_scene():
    return {
        "objects": [
            {"label": "red block",    "position": (2.5, 1.0)},
            {"label": "blue block",   "position": (3.0, 2.0)},
            {"label": "green block",  "position": (1.5, 3.0)},
            {"label": "yellow block", "position": (4.0, 2.5)},
            {"label": "left tray",    "position": (6.0, 1.0)},
            {"label": "right tray",   "position": (8.0, 1.0)},
            {"label": "workstation",  "position": (5.0, 5.0)},
        ]
    }

@pytest.fixture
def loaded_robot(sample_scene):
    robot = MockRobot()
    robot.load_scene(sample_scene)
    return robot

@pytest.fixture
def pick_instruction():
    return ParsedInstruction(
        action=ActionType.PICK,
        object_target="red block",
        destination=None,
        confidence=ConfidenceLevel.HIGH,
        raw_instruction="pick up the red block",
    )

@pytest.fixture
def place_instruction():
    return ParsedInstruction(
        action=ActionType.PLACE,
        object_target="blue block",
        destination="left tray",
        spatial_relation="in",
        confidence=ConfidenceLevel.HIGH,
        raw_instruction="place the blue block in the left tray",
    )

@pytest.fixture
def locate_instruction():
    return ParsedInstruction(
        action=ActionType.LOCATE,
        object_target="yellow block",
        confidence=ConfidenceLevel.HIGH,
        raw_instruction="locate the yellow block",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ACTION SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestActionSchema:

    def test_robot_command_valid(self):
        cmd = RobotCommand(
            step=1,
            command_type=CommandType.PICK,
            target_object="red block",
        )
        assert cmd.step == 1
        assert cmd.command_type == CommandType.PICK
        assert cmd.target_object == "red block"

    def test_position_as_tuple(self):
        pos = Position(x=2.5, y=1.0)
        assert pos.as_tuple() == (2.5, 1.0, 0.0)

    def test_action_plan_total_steps(self):
        plan = ActionPlan(
            instruction="pick up the red block",
            commands=[
                RobotCommand(step=1, command_type=CommandType.LOCATE, target_object="red block"),
                RobotCommand(step=2, command_type=CommandType.MOVE,   target_object="red block"),
                RobotCommand(step=3, command_type=CommandType.PICK,   target_object="red block"),
            ]
        )
        assert plan.total_steps == 3

    def test_command_summary(self):
        cmd = RobotCommand(
            step=1, command_type=CommandType.MOVE,
            target_object="red block",
            target_position=Position(x=2.5, y=1.0),
        )
        summary = cmd.summary()
        assert "Step 1" in summary
        assert "MOVE" in summary

    def test_plan_to_commands_simple(self):
        steps = [
            {"step": 1, "command": "locate", "target": "red block", "position": (2.5, 1.0)},
            {"step": 2, "command": "move",   "target": (2.5, 1.0)},
            {"step": 3, "command": "pick",   "target": "red block"},
        ]
        plan = plan_to_commands(steps, "pick up the red block")
        assert plan.total_steps == 3
        assert plan.commands[0].command_type == CommandType.LOCATE
        assert plan.commands[1].command_type == CommandType.MOVE
        assert plan.commands[2].command_type == CommandType.PICK

    def test_plan_to_commands_with_task_id(self):
        steps = [{"step": 1, "command": "locate", "target": "red block"}]
        plan  = plan_to_commands(steps, "test", task_id="abc123")
        assert plan.task_id == "abc123"

    def test_all_command_types_valid(self):
        for ct in CommandType:
            cmd = RobotCommand(step=1, command_type=ct)
            assert cmd.command_type == ct


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK ROBOT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMockRobot:

    def test_scene_loads_correctly(self, loaded_robot, sample_scene):
        obj_map = loaded_robot.get_object_map()
        assert len(obj_map) == len(sample_scene["objects"])
        assert "red block" in obj_map

    def test_move_to_valid_position(self, loaded_robot):
        result = loaded_robot.move_to(3.0, 2.0)
        assert result.success is True
        assert loaded_robot.get_position() == (3.0, 2.0)

    def test_move_to_out_of_bounds(self, loaded_robot):
        result = loaded_robot.move_to(99.0, 99.0)
        assert result.success is False
        assert "outside workspace" in result.message

    def test_move_to_object(self, loaded_robot):
        result = loaded_robot.move_to_object("red block")
        assert result.success is True
        assert loaded_robot.get_position() == (2.5, 1.0)

    def test_move_to_unknown_object(self, loaded_robot):
        result = loaded_robot.move_to_object("purple block")
        assert result.success is False

    def test_pick_success(self, loaded_robot):
        result = loaded_robot.pick("red block")
        assert result.success is True
        assert loaded_robot.get_held_object() == "red block"

    def test_pick_unknown_object(self, loaded_robot):
        result = loaded_robot.pick("purple block")
        assert result.success is False

    def test_pick_when_already_holding(self, loaded_robot):
        loaded_robot.pick("red block")
        result = loaded_robot.pick("blue block")
        assert result.success is False
        assert "already holding" in result.message.lower()

    def test_place_success(self, loaded_robot):
        loaded_robot.pick("red block")
        result = loaded_robot.place("left tray")
        assert result.success is True
        assert loaded_robot.get_held_object() is None

    def test_place_without_holding(self, loaded_robot):
        result = loaded_robot.place("left tray")
        assert result.success is False

    def test_place_unknown_location(self, loaded_robot):
        loaded_robot.pick("red block")
        result = loaded_robot.place("nonexistent tray")
        assert result.success is False

    def test_locate_success(self, loaded_robot):
        result = loaded_robot.locate("yellow block")
        assert result.success is True
        assert "4.0" in result.message or "4" in result.message

    def test_locate_unknown_object(self, loaded_robot):
        result = loaded_robot.locate("purple block")
        assert result.success is False

    def test_full_pick_place_sequence(self, loaded_robot):
        r1 = loaded_robot.move_to_object("red block")
        r2 = loaded_robot.pick("red block")
        r3 = loaded_robot.move_to_object("left tray")
        r4 = loaded_robot.place("left tray")
        assert all(r.success for r in [r1, r2, r3, r4])
        assert loaded_robot.get_held_object() is None

    def test_robot_reset(self, loaded_robot):
        loaded_robot.pick("red block")
        loaded_robot.reset()
        assert loaded_robot.get_held_object() is None
        assert loaded_robot.get_position() == (0.0, 0.0)
        assert loaded_robot.get_object_map() == {}

    def test_command_log_grows(self, loaded_robot):
        loaded_robot.move_to(1.0, 1.0)
        loaded_robot.pick("red block")
        assert len(loaded_robot.get_command_log()) == 2

    def test_case_insensitive_object_lookup(self, loaded_robot):
        result = loaded_robot.pick("RED BLOCK")
        assert result.success is True


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutor:

    def _make_plan(self, steps_data: list, instruction: str = "test") -> ActionPlan:
        return plan_to_commands(steps_data, instruction)

    def test_simple_locate_plan_succeeds(self, loaded_robot):
        plan = self._make_plan([
            {"step": 1, "command": "locate", "target": "red block"}
        ], "locate the red block")
        executor = Executor(loaded_robot)
        result   = executor.execute(plan, verbose=False)
        assert result.success is True
        assert result.steps_completed == 1

    def test_full_pick_place_plan_succeeds(self, loaded_robot):
        plan = self._make_plan([
            {"step": 1, "command": "locate", "target": "red block",  "position": (2.5, 1.0)},
            {"step": 2, "command": "move",   "target": "red block"},
            {"step": 3, "command": "pick",   "target": "red block"},
            {"step": 4, "command": "move",   "target": "left tray"},
            {"step": 5, "command": "place",  "target": "left tray"},
        ], "pick up the red block and place it in the left tray")
        executor = Executor(loaded_robot)
        result   = executor.execute(plan, verbose=False)
        assert result.success is True
        assert result.steps_completed == 5

    def test_plan_fails_on_unknown_object(self, loaded_robot):
        plan = self._make_plan([
            {"step": 1, "command": "pick", "target": "purple block"},
        ], "pick up the purple block")
        executor = Executor(loaded_robot)
        result   = executor.execute(plan, verbose=False)
        assert result.success is False
        assert result.failed_step == 1

    def test_plan_stops_on_first_failure(self, loaded_robot):
        plan = self._make_plan([
            {"step": 1, "command": "pick",  "target": "purple block"},  # fails
            {"step": 2, "command": "place", "target": "left tray"},      # should not run
        ], "test")
        executor = Executor(loaded_robot)
        result   = executor.execute(plan, verbose=False)
        assert result.steps_completed == 0
        assert result.failed_step == 1

    def test_execution_result_has_latency(self, loaded_robot):
        plan = self._make_plan([
            {"step": 1, "command": "locate", "target": "red block"}
        ], "test")
        executor = Executor(loaded_robot)
        result   = executor.execute(plan, verbose=False)
        assert result.total_latency_ms >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# TASK PLANNER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskPlanner:

    def test_pick_plan_generates_steps(self, pick_instruction, sample_scene):
        planner = TaskPlanner()
        plan    = planner.generate_plan(pick_instruction, sample_scene)
        assert plan.total_steps >= 3
        assert plan.commands[0].command_type == CommandType.LOCATE
        assert any(c.command_type == CommandType.PICK for c in plan.commands)

    def test_place_plan_generates_full_sequence(self, place_instruction, sample_scene):
        planner = TaskPlanner()
        plan    = planner.generate_plan(place_instruction, sample_scene)
        types   = [c.command_type for c in plan.commands]
        assert CommandType.LOCATE in types
        assert CommandType.PICK   in types
        assert CommandType.PLACE  in types

    def test_locate_plan_is_single_step(self, locate_instruction, sample_scene):
        planner = TaskPlanner()
        plan    = planner.generate_plan(locate_instruction, sample_scene)
        assert plan.total_steps == 1
        assert plan.commands[0].command_type == CommandType.LOCATE

    def test_pick_with_destination_includes_place(self, sample_scene):
        parsed = ParsedInstruction(
            action=ActionType.PICK,
            object_target="red block",
            destination="left tray",
            confidence=ConfidenceLevel.HIGH,
            raw_instruction="pick up the red block and place it in the left tray",
        )
        planner = TaskPlanner()
        plan    = planner.generate_plan(parsed, sample_scene)
        types   = [c.command_type for c in plan.commands]
        assert CommandType.PLACE in types

    def test_missing_object_raises(self, sample_scene):
        parsed = ParsedInstruction(
            action=ActionType.PICK,
            object_target="purple block",
            confidence=ConfidenceLevel.LOW,
            raw_instruction="pick up the purple block",
        )
        planner = TaskPlanner()
        with pytest.raises(ValueError, match="not found in scene"):
            planner.generate_plan(parsed, sample_scene)

    def test_plan_preserves_instruction(self, pick_instruction, sample_scene):
        planner = TaskPlanner()
        plan    = planner.generate_plan(pick_instruction, sample_scene)
        assert plan.instruction == pick_instruction.raw_instruction

    def test_plan_with_task_id(self, pick_instruction, sample_scene):
        planner = TaskPlanner()
        plan    = planner.generate_plan(pick_instruction, sample_scene, task_id="test123")
        assert plan.task_id == "test123"

    def test_move_plan_same_as_place(self, sample_scene):
        parsed = ParsedInstruction(
            action=ActionType.MOVE,
            object_target="green block",
            destination="right tray",
            confidence=ConfidenceLevel.HIGH,
            raw_instruction="move the green block to the right tray",
        )
        planner = TaskPlanner()
        plan    = planner.generate_plan(parsed, sample_scene)
        assert plan.total_steps >= 3

    def test_full_pipeline_planner_to_executor(self, place_instruction, sample_scene):
        """End-to-end: planner generates plan → executor runs it on mock robot."""
        planner  = TaskPlanner()
        robot    = MockRobot()
        robot.load_scene(sample_scene)
        executor = Executor(robot)

        plan   = planner.generate_plan(place_instruction, sample_scene)
        result = executor.execute(plan, verbose=False)

        assert result.success is True
        assert robot.get_held_object() is None  # object was placed


# ═══════════════════════════════════════════════════════════════════════════════
# SPATIAL RELATION & MULTI-STEP PLANNING TESTS
# ───────────────────────────────────────────────────────────────────────────────
# PITPP-61: Update unit tests for spatial relation and multi-step planning.
#
# These ten tests exercise the two capabilities added to the rule-based planner:
#
#   1. Spatial relations — the planner resolves "left of", "right of", "inside",
#      "on top of" and "near" into concrete drop coordinates relative to a
#      reference object (see task_planner.planner._apply_spatial_offset).
#
#   2. Multi-step planning — a single natural-language instruction is decomposed
#      into the canonical NINE-step primitive pipeline:
#
#          1. LOCATE  → 2. APPROACH → 3. LOWER → 4. GRASP → 5. LIFT
#          → 6. MOVE  → 7. LOWER    → 8. RELEASE → 9. RESET
#
# Every test mocks its two inputs deterministically — there are NO live LLM calls
# and NO PyBullet dependency:
#   • the LLM output is mocked as a hand-built ParsedInstruction (the input schema)
#   • the PyBullet scene is mocked as a dict of fixed coordinates
# This makes the suite fast, hermetic and suitable as regression evidence.
# ═══════════════════════════════════════════════════════════════════════════════

# The exact primitive sequence a full pick-and-place task must decompose into.
# Tests assert against this list so any drift in the planner is caught immediately.
EXPECTED_NINE_STEP_SEQUENCE = [
    PrimitiveActionType.LOCATE,    # 1. confirm the object is in the scene
    PrimitiveActionType.APPROACH,  # 2. hover above the object
    PrimitiveActionType.LOWER,     # 3. descend to grasp height
    PrimitiveActionType.GRASP,     # 4. close the gripper
    PrimitiveActionType.LIFT,      # 5. raise the object clear of the surface
    PrimitiveActionType.MOVE,      # 6. travel to the resolved destination
    PrimitiveActionType.LOWER,     # 7. descend to placement height
    PrimitiveActionType.RELEASE,   # 8. open the gripper to drop the object
    PrimitiveActionType.RESET,     # 9. return the arm to its home pose
]


class TestSpatialAndMultiStepPlanning:
    """Regression coverage for spatial-relation resolution and 9-step planning."""

    # ── Mocked inputs ───────────────────────────────────────────────────────────

    @pytest.fixture
    def mock_pybullet_scene(self):
        """
        Mock PyBullet scene data with fixed coordinates.

        In production these coordinates come from the vision/PyBullet layer. Here
        we stand in a MagicMock for that query layer and feed it deterministic
        results, so the planner is tested in complete isolation from PyBullet.
        """
        # A fake "world" the mocked query layer will look objects up in.
        world_coordinates = {
            "red block":   (2.5, 1.0),
            "blue block":  (3.0, 2.0),  # primary reference object for relations
            "green block": (1.5, 3.0),
            "left tray":   (6.0, 1.0),  # container used for "inside"
            "right tray":  (8.0, 1.0),
            "table":       (5.0, 5.0),  # flat surface used for "on top of"
        }

        # Mock the PyBullet accessor: pybullet_world.get_position(label) -> (x, y).
        pybullet_world = MagicMock(name="PyBulletWorld")
        pybullet_world.get_position.side_effect = lambda label: world_coordinates[label]

        # Assemble the scene dict the planner consumes, sourcing every coordinate
        # through the mock so the call is provably routed via the mocked layer.
        return {
            "objects": [
                {"label": label, "position": pybullet_world.get_position(label)}
                for label in world_coordinates
            ]
        }

    @staticmethod
    def _mock_parsed_instruction(
        object_target: str,
        destination: str | None = None,
        spatial_relation: str | None = None,
        action: ActionType = ActionType.PLACE,
        raw: str = "mocked instruction",
    ) -> ParsedInstruction:
        """
        Build a mocked LLM parse result (the planner's input schema).

        This substitutes for a real LLM call: we assert on planning behaviour,
        not on the LLM, so the parsed fields are supplied directly and
        deterministically.
        """
        return ParsedInstruction(
            action=action,
            object_target=object_target,
            destination=destination,
            spatial_relation=spatial_relation,
            confidence=ConfidenceLevel.HIGH,
            raw_instruction=raw,
        )

    def _primitive_actions(self, planner: TaskPlanner, parsed, scene) -> list:
        """Return the ordered list of PrimitiveActionType produced for an input."""
        planner_input = PlannerInput(parsed_instruction=parsed, scene=scene)
        plan = planner.generate_primitive_plan(planner_input)
        return [step.action for step in plan.steps]

    def _step_by_index(self, planner: TaskPlanner, parsed, scene, index: int):
        """Return the primitive step at a given position in the 9-step pipeline."""
        planner_input = PlannerInput(parsed_instruction=parsed, scene=scene)
        plan = planner.generate_primitive_plan(planner_input)
        return plan.steps[index]

    # ── 1–5: Spatial relation tests ──────────────────────────────────────────────

    def test_spatial_left_of_shifts_destination_negative_x(self, mock_pybullet_scene):
        """'left of' must place the object at the reference's X minus SPATIAL_OFFSET."""
        planner = TaskPlanner()
        parsed  = self._mock_parsed_instruction(
            "red block", destination="blue block", spatial_relation="left of",
            raw="place the red block to the left of the blue block",
        )
        # Step index 5 (0-based) is the MOVE-to-destination step (pipeline step 6).
        move_step = self._step_by_index(planner, parsed, mock_pybullet_scene, 5)

        # blue block is at x=3.0 → "left of" => 3.0 - 0.15 = 2.85, y unchanged.
        assert move_step.action == PrimitiveActionType.MOVE
        assert move_step.target_position.x == pytest.approx(3.0 - SPATIAL_OFFSET)
        assert move_step.target_position.y == pytest.approx(2.0)

    def test_spatial_right_of_shifts_destination_positive_x(self, mock_pybullet_scene):
        """'right of' must place the object at the reference's X plus SPATIAL_OFFSET."""
        planner = TaskPlanner()
        parsed  = self._mock_parsed_instruction(
            "red block", destination="blue block", spatial_relation="right of",
            raw="place the red block to the right of the blue block",
        )
        move_step = self._step_by_index(planner, parsed, mock_pybullet_scene, 5)

        # blue block at x=3.0 → "right of" => 3.0 + 0.15 = 3.15.
        assert move_step.target_position.x == pytest.approx(3.0 + SPATIAL_OFFSET)
        assert move_step.target_position.y == pytest.approx(2.0)

    def test_spatial_inside_keeps_container_centre(self, mock_pybullet_scene):
        """'inside' a container must drop at the container centre (no XY offset)."""
        planner = TaskPlanner()
        parsed  = self._mock_parsed_instruction(
            "red block", destination="left tray", spatial_relation="inside",
            raw="place the red block inside the left tray",
        )
        move_step = self._step_by_index(planner, parsed, mock_pybullet_scene, 5)

        # left tray at (6.0, 1.0) → "inside" leaves XY untouched.
        assert move_step.target_position.x == pytest.approx(6.0)
        assert move_step.target_position.y == pytest.approx(1.0)

    def test_spatial_on_top_of_raises_placement_height(self, mock_pybullet_scene):
        """'on top of' must lift the placement Z by STACK_HEIGHT to stack objects."""
        planner = TaskPlanner()
        parsed  = self._mock_parsed_instruction(
            "red block", destination="table", spatial_relation="on top of",
            raw="place the red block on top of the table",
        )
        # Step index 6 (0-based) is the LOWER-to-place step (pipeline step 7),
        # which carries the final placement height.
        lower_step = self._step_by_index(planner, parsed, mock_pybullet_scene, 6)

        assert lower_step.action == PrimitiveActionType.LOWER
        # XY stay at the table centre, Z is raised to stack on top.
        assert lower_step.target_position.x == pytest.approx(5.0)
        assert lower_step.target_position.y == pytest.approx(5.0)
        assert lower_step.target_position.z == pytest.approx(STACK_HEIGHT)

    def test_spatial_near_applies_small_offset(self, mock_pybullet_scene):
        """'near' must apply the smaller NEAR_OFFSET, not the full SPATIAL_OFFSET."""
        planner = TaskPlanner()
        parsed  = self._mock_parsed_instruction(
            "red block", destination="blue block", spatial_relation="near",
            raw="place the red block near the blue block",
        )
        move_step = self._step_by_index(planner, parsed, mock_pybullet_scene, 5)

        # blue block at x=3.0 → "near" => 3.0 + 0.10 (NEAR_OFFSET).
        assert move_step.target_position.x == pytest.approx(3.0 + NEAR_OFFSET)
        # And it must be a genuinely smaller shift than the lateral offset.
        assert NEAR_OFFSET < SPATIAL_OFFSET

    # ── 6–10: Multi-step planning tests ──────────────────────────────────────────

    def test_multistep_decomposes_into_exact_nine_step_sequence(self, mock_pybullet_scene):
        """A full pick-and-place must decompose into the exact 9-step pipeline."""
        planner = TaskPlanner()
        parsed  = self._mock_parsed_instruction(
            "red block", destination="left tray", spatial_relation="inside",
            raw="pick up the red block and place it inside the left tray",
        )
        actions = self._primitive_actions(planner, parsed, mock_pybullet_scene)

        # Exact ordered match against Locate → … → Reset.
        assert actions == EXPECTED_NINE_STEP_SEQUENCE
        assert len(actions) == 9

    def test_multistep_command_plan_maps_to_executor_commands(self, mock_pybullet_scene):
        """The 9 primitives must map to the correct executor-level command types."""
        planner = TaskPlanner()
        parsed  = self._mock_parsed_instruction(
            "blue block", destination="right tray", spatial_relation="inside",
            raw="move the blue block into the right tray",
        )
        plan = planner.generate_plan(parsed, mock_pybullet_scene)

        # GRASP→PICK, RELEASE→PLACE, every motion (incl. RESET)→MOVE, LOCATE→LOCATE.
        expected_command_types = [
            CommandType.LOCATE,  # 1. locate
            CommandType.MOVE,    # 2. approach
            CommandType.MOVE,    # 3. lower
            CommandType.PICK,    # 4. grasp
            CommandType.MOVE,    # 5. lift
            CommandType.MOVE,    # 6. move to destination
            CommandType.MOVE,    # 7. lower to place
            CommandType.PLACE,   # 8. release
            CommandType.MOVE,    # 9. reset to home
        ]
        assert plan.total_steps == 9
        assert [c.command_type for c in plan.commands] == expected_command_types
        # Steps must be numbered 1..9 in order.
        assert [c.step for c in plan.commands] == list(range(1, 10))

    def test_multistep_sequence_ends_with_reset_to_home(self, mock_pybullet_scene):
        """The pipeline must always terminate with a RESET to the home pose."""
        planner = TaskPlanner()
        parsed  = self._mock_parsed_instruction(
            "green block", destination="left tray", spatial_relation="inside",
            raw="put the green block in the left tray",
        )
        planner_input = PlannerInput(parsed_instruction=parsed, scene=mock_pybullet_scene)
        steps = planner.generate_primitive_plan(planner_input).steps

        last = steps[-1]
        assert last.action == PrimitiveActionType.RESET
        # RESET returns the arm to the configured home position.
        assert last.target_position.x == pytest.approx(HOME_POSITION.x)
        assert last.target_position.y == pytest.approx(HOME_POSITION.y)
        assert last.target_position.z == pytest.approx(HOME_POSITION.z)

    def test_multistep_executes_end_to_end_on_mock_robot(self, mock_pybullet_scene):
        """The full 9-step spatial plan must execute successfully on the robot."""
        planner  = TaskPlanner()
        robot    = MockRobot()
        robot.load_scene(mock_pybullet_scene)
        executor = Executor(robot)

        parsed = self._mock_parsed_instruction(
            "red block", destination="blue block", spatial_relation="left of",
            raw="pick up the red block and place it to the left of the blue block",
        )
        plan   = planner.generate_plan(parsed, mock_pybullet_scene)
        result = executor.execute(plan, verbose=False)

        # All nine steps run, the object is released, and the arm ends empty-handed.
        assert result.success is True
        assert result.steps_completed == 9
        assert robot.get_held_object() is None

    def test_multistep_obstacle_relation_changes_only_destination(self, mock_pybullet_scene):
        """
        Re-planning the same task around an obstacle (a different spatial relation)
        must keep the 9-step structure and the grasp pose identical, changing only
        the resolved destination — proving the decomposition is relation-stable.
        """
        planner = TaskPlanner()

        # Same object & reference, two different spatial relations (e.g. routing
        # the object to the other side of an obstacle).
        parsed_left = self._mock_parsed_instruction(
            "red block", destination="blue block", spatial_relation="left of",
            raw="place the red block left of the blue block",
        )
        parsed_right = self._mock_parsed_instruction(
            "red block", destination="blue block", spatial_relation="right of",
            raw="place the red block right of the blue block",
        )

        # Structure is identical for both plans (same 9 primitives in order).
        assert (
            self._primitive_actions(planner, parsed_left, mock_pybullet_scene)
            == self._primitive_actions(planner, parsed_right, mock_pybullet_scene)
            == EXPECTED_NINE_STEP_SEQUENCE
        )

        # The grasp-side APPROACH (step index 1) is unchanged — same pick pose...
        approach_left  = self._step_by_index(planner, parsed_left,  mock_pybullet_scene, 1)
        approach_right = self._step_by_index(planner, parsed_right, mock_pybullet_scene, 1)
        assert approach_left.target_position.x == pytest.approx(approach_right.target_position.x)

        # ...while only the destination MOVE (step index 5) differs by 2*offset.
        move_left  = self._step_by_index(planner, parsed_left,  mock_pybullet_scene, 5)
        move_right = self._step_by_index(planner, parsed_right, mock_pybullet_scene, 5)
        assert move_right.target_position.x - move_left.target_position.x == pytest.approx(
            2 * SPATIAL_OFFSET
        )
