"""
display_scene.py
----------------
Standalone visual debugging tool for the simulation environment.

Launches the PyBullet GUI window and an OpenCV camera feed window
showing the current simulation scene state. Nothing from the main
pipeline is involved — this tool is purely for visual inspection.

ONLY runnable as a module from the project root:
    python -m simulation_backend.display_scene

Do NOT import or call this from simulation.py or any other pipeline file.

What is shown:
    PyBullet GUI window:
        - Full 3D physics simulation with all scene objects
        - Floating object labels above each object (configurable)
        - Robot state debug text overlay (configurable)

    OpenCV camera feed window:
        - Live camera frame from the configured eye-to-hand camera
        - Detection bounding boxes overlaid (if VISION_DETECTOR is set)
        - Segmentation mask overlay (if SEGMENTATION_MASK=true)

    Terminal output:
        - Current scene dict printed at configured rate
        - Robot state (position, held object) printed at configured rate

Press Q in either window or Ctrl+C in terminal to exit.

Config:
    All display settings are in scene_config.yaml under the 'display' key.
    Object labels can be disabled by setting display.object_labels.enabled: false
"""

import os
import sys
import time
import logging
import signal

import pybullet as p
import numpy as np
import cv2
import yaml

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Config path ────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "scene_config.yaml"
)

# ── Globals ────────────────────────────────────────────────────────────────────
_running = True


def _signal_handler(sig, frame):
    global _running
    logger.info("Shutdown signal received.")
    _running = False


signal.signal(signal.SIGINT, _signal_handler)


# ── Config loader ──────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── PyBullet setup ─────────────────────────────────────────────────────────────

def _connect_gui() -> int:
    client = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setRealTimeSimulation(1, physicsClientId=client)
    logger.info("PyBullet GUI connected.")
    return client


# ── Object label overlays ──────────────────────────────────────────────────────

def _add_object_labels(
    client:   int,
    registry,
    label_cfg: dict,
) -> list[int]:
    """
    Add floating debug text labels above each object in the PyBullet GUI.

    Returns list of debug item IDs so they can be removed if needed.
    Controlled by display.object_labels.enabled in scene_config.yaml.
    """
    if not label_cfg.get("enabled", True):
        logger.info("Object labels disabled in config.")
        return []

    color     = label_cfg.get("color", [1.0, 1.0, 1.0])
    size      = label_cfg.get("size", 1.2)
    offset_z  = label_cfg.get("offset_z", 0.15)

    debug_ids = []
    for entry in registry.all_entries():
        x, y, z = entry.position
        text_pos = [x, y, z + offset_z]

        debug_id = p.addUserDebugText(
            text=entry.label,
            textPosition=text_pos,
            textColorRGB=color,
            textSize=size,
            physicsClientId=client,
        )
        debug_ids.append(debug_id)

    logger.info(f"Added {len(debug_ids)} object labels to PyBullet GUI.")
    return debug_ids


def _remove_object_labels(client: int, debug_ids: list[int]) -> None:
    """Remove all object label debug items from the PyBullet GUI."""
    for debug_id in debug_ids:
        p.removeUserDebugItem(debug_id, physicsClientId=client)


# ── Robot state overlay ────────────────────────────────────────────────────────

def _update_robot_state_overlay(
    client:     int,
    overlay_id: int | None,
    state_text: str,
    overlay_cfg: dict,
) -> int:
    """
    Update the robot state debug text in the PyBullet GUI.

    Creates or replaces the debug text item each frame.
    Returns the new debug item ID.
    """
    if not overlay_cfg.get("enabled", True):
        return overlay_id

    if overlay_id is not None:
        p.removeUserDebugItem(overlay_id, physicsClientId=client)

    position = overlay_cfg.get("position", [0.0, 0.0, 1.5])
    color    = overlay_cfg.get("color", [0.0, 1.0, 0.5])
    size     = overlay_cfg.get("size", 1.0)

    new_id = p.addUserDebugText(
        text=state_text,
        textPosition=position,
        textColorRGB=color,
        textSize=size,
        physicsClientId=client,
    )
    return new_id


# ── Camera capture ─────────────────────────────────────────────────────────────

def _capture_frame(client: int, cam_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Capture a frame from the configured camera.

    Returns:
        (rgb_frame, depth_buffer, seg_mask) as numpy arrays
    """
    position  = cam_cfg.get("position", [1.5, 0.0, 1.8])
    target    = cam_cfg.get("target",   [0.0, 0.0, 0.0])
    up_vector = cam_cfg.get("up_vector", [0.0, 0.0, 1.0])
    fov       = cam_cfg.get("fov", 60)
    aspect    = cam_cfg.get("aspect_ratio", 1.333)
    near      = cam_cfg.get("near_val", 0.1)
    far       = cam_cfg.get("far_val",  10.0)
    width     = cam_cfg.get("width",  640)
    height    = cam_cfg.get("height", 480)

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=position,
        cameraTargetPosition=target,
        cameraUpVector=up_vector,
        physicsClientId=client,
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=fov, aspect=aspect,
        nearVal=near, farVal=far,
        physicsClientId=client,
    )

    width_ret, height_ret, rgba, depth, seg = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        physicsClientId=client,
    )

    # Reshape and convert
    rgba_img  = np.reshape(rgba,  (height, width, 4)).astype(np.uint8)
    rgb_img   = rgba_img[:, :, :3]
    bgr_img   = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    depth_arr = np.reshape(depth, (height, width)).astype(np.float32)
    seg_arr   = np.reshape(seg,   (height, width)).astype(np.int32)

    return bgr_img, depth_arr, seg_arr


# ── Segmentation overlay ───────────────────────────────────────────────────────

def _overlay_segmentation(
    frame:    np.ndarray,
    seg_mask: np.ndarray,
    registry,
) -> np.ndarray:
    """
    Overlay segmentation mask on the OpenCV frame.
    Each object gets a semi-transparent colour overlay using its
    registered colour.
    """
    overlay = frame.copy()

    for entry in registry.all_entries():
        mask = (seg_mask == entry.body_id)
        if not mask.any():
            continue

        r, g, b = entry.color[0], entry.color[1], entry.color[2]
        overlay[mask] = [int(b * 255), int(g * 255), int(r * 255)]

    # Blend with original
    return cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)


# ── Terminal output ────────────────────────────────────────────────────────────

def _print_scene(scene: dict) -> None:
    """Print current scene dict to terminal."""
    print("\n─── Scene State ───────────────────────────────────────────")
    for obj in scene.get("detected_objects", []):
        coords = obj["position"]["coordinates_3d"]
        print(
            f"  {obj['label']:<20} "
            f"x={coords['x']:>6.3f}  y={coords['y']:>6.3f}  z={coords['z']:>6.3f}  "
            f"source={obj['source']}"
        )
    print(f"  Timestamp: {scene.get('scene_timestamp', 'unknown')}")
    print("───────────────────────────────────────────────────────────")


# ── Main display loop ──────────────────────────────────────────────────────────

def run() -> None:
    global _running

    cfg         = _load_config()
    cam_cfg     = cfg.get("camera", {})
    display_cfg = cfg.get("display", {})
    ws_cfg      = cfg.get("workspace", {})
    obj_cfg     = cfg.get("objects", [])

    label_cfg   = display_cfg.get("object_labels", {})
    overlay_cfg = display_cfg.get("robot_state_overlay", {})
    cv_cfg      = display_cfg.get("opencv_window", {})
    term_cfg    = display_cfg.get("terminal", {})

    # ── Connect to PyBullet GUI ────────────────────────────────────────────────
    client = _connect_gui()

    # ── Build workspace and load objects ───────────────────────────────────────
    from simulation_backend.simulation_environment.workspace import Workspace
    from simulation_backend.simulation_environment.object_loader import ObjectLoader
    from simulation_backend.simulation_environment.object_registry import ObjectRegistry

    registry  = ObjectRegistry()
    workspace = Workspace(client, ws_cfg)
    workspace.build()

    loader = ObjectLoader(client, obj_cfg, registry)
    loader.load_all()

    # ── Add object labels to GUI ───────────────────────────────────────────────
    label_ids = _add_object_labels(client, registry, label_cfg)

    # ── Build initial scene for display ───────────────────────────────────────
    from simulation_backend.simulation_environment.scene_builder import SceneBuilder
    from simulation_backend.vision.ground_truth import GroundTruth

    ground_truth = GroundTruth(client, registry)
    builder      = SceneBuilder(registry)

    # ── Segmentation setup ─────────────────────────────────────────────────────
    seg_enabled = os.getenv("SEGMENTATION_MASK", "false").lower() == "true"
    if seg_enabled:
        from simulation_backend.vision.segmentation import Segmentation
        segmentation = Segmentation(client, registry)
        logger.info("Segmentation mask enabled.")
    else:
        segmentation = None

    # ── OpenCV window setup ────────────────────────────────────────────────────
    cv_enabled   = cv_cfg.get("enabled", True)
    window_name  = cv_cfg.get("window_name", "Simulation Camera Feed")
    cv_rate      = cv_cfg.get("update_rate_hz", 30)
    if cv_enabled:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ── Timing ────────────────────────────────────────────────────────────────
    term_rate         = term_cfg.get("update_rate_hz", 2)
    last_cv_time      = 0.0
    last_term_time    = 0.0
    robot_overlay_id  = None

    logger.info("Display scene running. Press Q in camera window or Ctrl+C to exit.")
    print("\n" + "=" * 60)
    print("  simulation_backend — Display Scene")
    print(f"  Objects loaded: {len(registry)}")
    print(f"  Segmentation:   {seg_enabled}")
    print("  Press Q in the camera window or Ctrl+C to exit.")
    print("=" * 60 + "\n")

    while _running:
        now = time.time()
        p.stepSimulation(physicsClientId=client)

        # ── OpenCV update ──────────────────────────────────────────────────────
        if cv_enabled and (now - last_cv_time) >= 1.0 / cv_rate:
            frame, depth, seg_mask = _capture_frame(client, cam_cfg)

            # Segmentation overlay
            if seg_enabled and cv_cfg.get("show_segmentation", True):
                frame = _overlay_segmentation(frame, seg_mask, registry)

            cv2.imshow(window_name, frame)
            last_cv_time = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Q pressed — exiting.")
                _running = False
                break

        # ── Robot state overlay update ─────────────────────────────────────────
        if overlay_cfg.get("enabled", True):
            state_text = "Robot: idle | Held: None | Pos: (0.0, 0.0, 0.0)"
            robot_overlay_id = _update_robot_state_overlay(
                client, robot_overlay_id, state_text, overlay_cfg
            )

        # ── Terminal update ────────────────────────────────────────────────────
        if (term_cfg.get("enabled", True)
                and (now - last_term_time) >= 1.0 / term_rate):

            gt_detections = ground_truth.get_all()
            scene = builder.build(
                detector_results=[],
                segmentation_results=None,
                ground_truth_results=gt_detections,
            )

            if term_cfg.get("print_scene", True):
                _print_scene(scene)

            last_term_time = now

        time.sleep(1.0 / 240.0)

    # ── Cleanup ────────────────────────────────────────────────────────────────
    logger.info("Cleaning up...")
    if cv_enabled:
        cv2.destroyAllWindows()
    if label_ids:
        _remove_object_labels(client, label_ids)
    p.disconnect(physicsClientId=client)
    logger.info("Display scene exited cleanly.")


if __name__ == "__main__":
    run()
