from vision_backend.scene_representation import get_planner_scene
from vision_backend.invalid_actions import handle_invalid_action

scene = get_planner_scene()

handle_invalid_action(
    scene=scene,
    object_name="yellow block",
    instruction="pick up the yellow block"
)