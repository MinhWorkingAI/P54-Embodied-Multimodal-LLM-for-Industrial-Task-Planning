"""
main.py
-------
Wires all modules together:
    User instruction
        → LLM parse
        → Vision lookup
        → Task plan
        → Execution
        → Feedback

Now integrated with:
    ✓ Real PyBullet robot
    ✓ RealRobot execution
    ✓ Physical objects
"""

import os
import sys
import argparse
import logging
import time

import pybullet as p
import pybullet_data

os.environ["PYDANTIC_DISABLE_PLUGINS"] = "1"

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# ── Module imports ─────────────────────────────────────────────────────────────
from llm_backend.custom_LLM_parser import parse_instruction
from llm_backend.schema import ConfidenceLevel
from llm_backend.tracker import PipelineTracker

from task_planner.planner import TaskPlanner

from simulation_backend.real_robot import RealRobot
from simulation_backend.executor import Executor

from simulation_backend.simulation_environment.object_registry import ObjectRegistry

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)

SEP = "═" * 60


# ── Scene (TEMP STUB) ─────────────────────────────────────────────────────────

DEFAULT_SCENE = {
    "objects": [
        {"label": "red block", "position": (0.5, 0.2, 0.02)},
        {"label": "blue block", "position": (0.6, -0.2, 0.02)},
        {"label": "left tray", "position": (0.3, 0.3, 0.02)},
        {"label": "right tray", "position": (0.7, -0.3, 0.02)},
    ]
}


def get_scene():
    return DEFAULT_SCENE


# ── PYBULLET SETUP ────────────────────────────────────────────────────────────

def setup_simulation():

    physics_client = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")

    # Load robot
    robot_id = p.loadURDF(
        "kuka_iiwa/model.urdf",
        basePosition=[0, 0, 0],
        useFixedBase=True
    )

    # Create registry
    object_registry = ObjectRegistry()

    # Load objects
    red_block = p.loadURDF(
        "cube_small.urdf",
        [0.5, 0.2, 0.02]
    )

    blue_block = p.loadURDF(
        "cube_small.urdf",
        [0.6, -0.2, 0.02]
    )

    # Register objects
    object_registry.register_object(
        "red block",
        red_block
    )

    object_registry.register_object(
        "blue block",
        blue_block
    )

    return robot_id, object_registry


# ── PIPELINE ──────────────────────────────────────────────────────────────────

def run_pipeline(
    instruction: str,
    verbose: bool = True,
    tracker: PipelineTracker | None = None,
):

    if tracker is None:
        tracker = PipelineTracker()

    backend = os.getenv("LLM_BACKEND", "openai")

    task_id = tracker.new_task(
        instruction,
        model=backend
    )

    if verbose:
        print(f"\n{SEP}")
        print("PIPELINE START")
        print(f"Instruction : {instruction}")
        print(f"Backend     : {backend}")
        print(f"Task ID     : {task_id}")
        print(SEP)

    result = {
        "success": False,
        "task_id": task_id,
    }

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1 — LLM PARSE
    # ═══════════════════════════════════════════════════════════════

    if verbose:
        print("\n[1/5] LLM Parse")

    try:

        t0 = time.perf_counter()

        parsed = parse_instruction(instruction)

        latency = (time.perf_counter() - t0) * 1000

        tracker.record(
            task_id,
            "llm_parse",
            status="success",
            payload=parsed.model_dump(mode="json"),
            latency_ms=latency
        )

        if verbose:
            print(f"Action      : {parsed.action.value}")
            print(f"Object      : {parsed.object_target}")
            print(f"Destination : {parsed.destination}")
            print(f"Confidence  : {parsed.confidence.value}")

        if parsed.confidence == ConfidenceLevel.LOW:

            print("\nLow confidence instruction")
            print(parsed.notes)

            tracker.complete_task(task_id, success=False)

            return result

    except Exception as e:

        print(f"LLM Parse Error: {e}")

        tracker.complete_task(task_id, success=False)

        return result

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2 — SCENE LOOKUP
    # ═══════════════════════════════════════════════════════════════

    if verbose:
        print("\n[2/5] Scene Lookup")

    try:

        scene = get_scene()

        if verbose:
            print("Scene Objects:")

            for obj in scene["objects"]:
                print(f" - {obj['label']}")

    except Exception as e:

        print(f"Scene Error: {e}")

        tracker.complete_task(task_id, success=False)

        return result

    # ═══════════════════════════════════════════════════════════════
    # STAGE 3 — TASK PLANNING
    # ═══════════════════════════════════════════════════════════════

    if verbose:
        print("\n[3/5] Task Planning")

    try:

        planner = TaskPlanner()

        plan = planner.generate_plan(
            parsed,
            scene,
            task_id=task_id
        )

        if verbose:

            print(f"Generated {plan.total_steps} steps")

            for cmd in plan.commands:
                print(cmd.summary())

    except Exception as e:

        print(f"Planning Error: {e}")

        tracker.complete_task(task_id, success=False)

        return result

    # ═══════════════════════════════════════════════════════════════
    # STAGE 4 — REAL ROBOT EXECUTION
    # ═══════════════════════════════════════════════════════════════

    if verbose:
        print("\n[4/5] Real Robot Execution")

    try:

        robot_id, object_registry = setup_simulation()

        robot = RealRobot(
            robot_id=robot_id,
            object_registry=object_registry
        )

        robot.load_scene(scene)

        executor = Executor(
            robot,
            tracker=tracker,
            task_id=task_id
        )

        execution_result = executor.execute(
            plan,
            verbose=verbose
        )

        result["execution"] = execution_result

    except Exception as e:

        print(f"Execution Error: {e}")

        tracker.complete_task(task_id, success=False)

        return result

    # ═══════════════════════════════════════════════════════════════
    # STAGE 5 — FEEDBACK
    # ═══════════════════════════════════════════════════════════════

    if verbose:
        print("\n[5/5] Feedback")

        print("Task completed successfully")

    tracker.complete_task(task_id, success=True)

    result["success"] = True

    return result


# ── INTERACTIVE MODE ──────────────────────────────────────────────────────────

def run_interactive():

    tracker = PipelineTracker()

    print(f"\n{SEP}")
    print("Industrial Task Planning Pipeline")
    print("Type 'quit' to exit")
    print(SEP)

    while True:

        try:

            instruction = input("\nInstruction: ").strip()

            if instruction.lower() in ["quit", "exit", "q"]:
                break

            if not instruction:
                continue

            run_pipeline(
                instruction,
                verbose=True,
                tracker=tracker
            )

        except KeyboardInterrupt:
            break

        except Exception as e:
            print(f"Pipeline Error: {e}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "instruction",
        nargs="?",
        help="Instruction to execute"
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true"
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true"
    )

    args = parser.parse_args()

    if args.interactive:

        run_interactive()

    elif args.instruction:

        run_pipeline(
            args.instruction,
            verbose=not args.quiet
        )

    else:

        parser.print_help()