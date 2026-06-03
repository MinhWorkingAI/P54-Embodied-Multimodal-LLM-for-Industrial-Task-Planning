"""
simulation_backend/simulation.py
---------------------------------
Simulation orchestrator for the P54 pipeline.

Owns the full PyBullet session lifecycle:
    - connects PyBullet (DIRECT for pipeline, GUI for visual debug)
    - applies physics settings from scene_config.yaml
    - builds the workspace (table, floor, walls)
    - loads all scene objects via ObjectLoader → ObjectRegistry
    - creates Camera, GroundTruth, SceneBuilder
    - optionally creates a primary detector via get_detector()

Public interface:
    get_live_scene() → dict
        Captures one camera frame, runs detector (if active) and ground
        truth fallback, returns the planner-compatible scene dict:
        {"objects": [{"label": str, "position": (x, y)}, ...]}

    get_robot() → MockRobot
        Returns the active robot instance ready for Executor.
        Phase 1: always MockRobot. Phase 3: real PyBullet robot.

    reset() → None
        Resets all object positions to their scene_config.yaml defaults.
        Call between pipeline runs so the scene is fresh each time.

    disconnect() → None
        Cleanly disconnects from PyBullet. Always call this on exit.

Usage (from main.py):
    from simulation_backend.simulation import Simulation

    sim = Simulation()              # reads scene_config.yaml automatically
    scene = sim.get_live_scene()    # Stage 2: live vision lookup
    robot = sim.get_robot()         # Stage 4: robot instance
    robot.load_scene(scene)
    ...
    sim.reset()                     # between instructions if needed
    sim.disconnect()                # on exit

Environment variables:
    USE_LIVE_SIMULATION=true    enable this class (false → JSON fallback)
    VISION_DETECTOR=colour      activate a primary detector (optional)
    SIMULATION_MODE=DIRECT      DIRECT (headless) or GUI (visual window)
"""

import os
import logging
import yaml
import pybullet as p
import pybullet_data

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "simulation_backend",
    "scene_config.yaml",
)

# Fallback: try relative to project root if the above doesn't exist
_CONFIG_PATH_ALT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scene_config.yaml",
)


class Simulation:
    """
    Manages the full PyBullet simulation session for the pipeline.

    Phase 1 capabilities:
        - Live scene via GroundTruth (exact PyBullet positions)
        - Optional primary detector (set VISION_DETECTOR in .env)
        - MockRobot for execution (no real arm yet)

    Phase 3 will add:
        - Real robot URDF loading
        - RobotBase subclass selection via ROBOT_MODEL env var
    """

    def __init__(
        self,
        config_path: str = None,
        mode: str = None,
    ) -> None:
        """
        Connect to PyBullet and build the full simulation scene.

        Args:
            config_path : Path to scene_config.yaml.
                          Defaults to simulation_backend/scene_config.yaml.
            mode        : "DIRECT" (headless, for pipeline) or "GUI" (window).
                          Overridden by SIMULATION_MODE env var if set.
                          Defaults to "DIRECT".
        """
        self._cfg        = self._load_config(config_path)
        self._client     = self._connect(mode)
        self._registry   = None
        self._loader     = None
        self._workspace  = None
        self._camera     = None
        self._ground_truth = None
        self._builder    = None
        self._detector   = None
        self._robot      = None

        self._build_scene()
        logger.info("Simulation initialised successfully.")

    # ── Public interface ───────────────────────────────────────────────────────

    def get_live_scene(self) -> dict:
        """
        Capture the current state of the scene and return the planner dict.

        Runs the full detection stack:
            1. Camera.capture() → CameraFrame
            2. Primary detector (if VISION_DETECTOR is set) → list[Detection]
            3. GroundTruth.get_all() → list[Detection] (always, as fallback)
            4. SceneBuilder.build() → rich scene dict
            5. SceneBuilder.to_planner_format() → {"objects": [...]}

        Returns:
            {
                "objects": [
                    {"label": "red block",  "position": (0.45, -0.20)},
                    {"label": "blue block", "position": (0.60,  0.10)},
                    ...
                ]
            }
        """
        # Step the simulation once to settle physics before reading positions
        p.stepSimulation(physicsClientId=self._client)

        # Primary detector results (empty list if no detector configured)
        detector_results = []
        if self._detector is not None:
            try:
                frame = self._camera.capture()
                detector_results = self._detector.detect(frame)
                logger.debug(f"Detector returned {len(detector_results)} detections.")
            except Exception as e:
                logger.warning(f"Detector failed, falling back to ground truth: {e}")

        # Ground truth fallback — always run
        gt_results = self._ground_truth.get_all()

        # Assemble rich scene, then convert to planner format
        rich_scene = self._builder.build(
            detector_results=detector_results,
            segmentation_results=None,
            ground_truth_results=gt_results,
        )

        planner_scene = self._builder.to_planner_format(rich_scene)
        logger.debug(
            f"Live scene: {len(planner_scene['objects'])} objects "
            f"(source: {'detector' if detector_results else 'ground_truth'})"
        )
        return planner_scene

    def get_robot(self):
        """
        Return the active robot instance.

        Phase 1: always MockRobot (no PyBullet arm, pure Python state machine).
        Phase 3: returns a RobotBase subclass (FrankaPanda, KukaIIWA, UR5)
                 selected by ROBOT_MODEL env var.

        Returns:
            MockRobot instance (Phase 1).
        """
        return self._robot

    def reset(self) -> None:
        """
        Reset all scene objects to their original positions from scene_config.yaml.

        Call this between pipeline runs so a pick-and-place from the previous
        instruction doesn't affect the starting state of the next one.
        """
        if self._loader is not None:
            self._loader.reset_positions()
            logger.info("Scene reset to initial positions.")

        # Reset robot state too
        if self._robot is not None:
            self._robot.reset()

    def disconnect(self) -> None:
        """
        Disconnect from PyBullet. Always call on exit to free resources.
        """
        try:
            p.disconnect(physicsClientId=self._client)
            logger.info("PyBullet disconnected.")
        except Exception as e:
            logger.warning(f"Disconnect error (may already be disconnected): {e}")

    @property
    def registry(self):
        """Return the ObjectRegistry (read-only access for debugging)."""
        return self._registry

    @property
    def client(self) -> int:
        """Return the PyBullet client ID."""
        return self._client

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _connect(self, mode: str = None) -> int:
        """Connect to PyBullet and apply physics settings."""
        env_mode = os.getenv("SIMULATION_MODE", "DIRECT").upper()
        resolved_mode = (mode or env_mode).upper()

        pybullet_mode = p.GUI if resolved_mode == "GUI" else p.DIRECT
        client = p.connect(pybullet_mode)

        physics_cfg = self._cfg.get("physics", {})
        gravity = physics_cfg.get("gravity", [0.0, 0.0, -9.81])
        p.setGravity(*gravity, physicsClientId=client)

        timestep = physics_cfg.get("timestep", 0.00416)
        p.setTimeStep(timestep, physicsClientId=client)

        solver_iter = physics_cfg.get("solver_iterations", 150)
        p.setPhysicsEngineParameter(
            numSolverIterations=solver_iter,
            physicsClientId=client,
        )

        # pybullet_data provides plane.urdf for the floor
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)

        logger.info(
            f"PyBullet connected — mode={resolved_mode} "
            f"client={client} gravity={gravity}"
        )
        return client

    def _build_scene(self) -> None:
        """Build workspace, load objects, create camera and vision stack."""
        from simulation_backend.simulation_environment.workspace import Workspace
        from simulation_backend.simulation_environment.object_loader import ObjectLoader
        from simulation_backend.simulation_environment.object_registry import ObjectRegistry
        from simulation_backend.simulation_environment.scene_builder import SceneBuilder
        from simulation_backend.vision.ground_truth import GroundTruth
        from simulation_backend.vision.camera import Camera
        from simulation_backend.vision.detection_base import get_detector
        from simulation_backend.mock_robot import MockRobot

        ws_cfg  = self._cfg.get("workspace", {})
        obj_cfg = self._cfg.get("objects", [])
        cam_cfg = self._cfg.get("camera", {})

        # 1. Object registry — populated by loader below
        self._registry = ObjectRegistry()

        # 2. Workspace — floor, table, walls
        self._workspace = Workspace(self._client, ws_cfg)
        self._workspace.build()
        logger.info("Workspace built.")

        # 3. Objects — all entries in scene_config.yaml objects list
        self._loader = ObjectLoader(self._client, obj_cfg, self._registry)
        self._loader.load_all()
        logger.info(f"Loaded {len(self._registry)} objects into registry.")

        # 4. Camera — eye-to-hand, fixed mount
        self._camera = Camera(physics_client=self._client, config=cam_cfg)
        logger.info(f"Camera ready: {self._camera}")

        # 5. Ground truth — exact PyBullet positions, always available
        self._ground_truth = GroundTruth(self._client, self._registry)

        # 6. Scene builder — assembles rich scene + converts to planner format
        self._builder = SceneBuilder(self._registry)

        # 7. Primary detector (optional — VISION_DETECTOR in .env)
        self._detector = get_detector(registry=self._registry, config=self._cfg)
        if self._detector:
            logger.info(f"Primary detector active: {self._detector}")
        else:
            logger.info("No primary detector — using ground truth only.")

        # 8. Robot — Phase 1: always MockRobot
        self._robot = MockRobot()
        logger.info("Robot: MockRobot (Phase 1 — no real arm).")

    @staticmethod
    def _load_config(config_path: str = None) -> dict:
        """Load and return scene_config.yaml as a dict."""
        if config_path and os.path.exists(config_path):
            path = config_path
        elif os.path.exists(_CONFIG_PATH):
            path = _CONFIG_PATH
        elif os.path.exists(_CONFIG_PATH_ALT):
            path = _CONFIG_PATH_ALT
        else:
            # Search upward from current file for scene_config.yaml
            search_dirs = [
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_backend"),
                os.getcwd(),
                os.path.join(os.getcwd(), "simulation_backend"),
            ]
            path = None
            for d in search_dirs:
                candidate = os.path.join(d, "scene_config.yaml")
                if os.path.exists(candidate):
                    path = candidate
                    break

            if path is None:
                raise FileNotFoundError(
                    "scene_config.yaml not found. "
                    "Run from the project root or pass config_path explicitly."
                )

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        logger.info(f"Config loaded from: {path}")
        return cfg
