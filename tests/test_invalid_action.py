from drafts.vision_backend.scene_representation import get_planner_scene
from drafts.vision_backend.invalid_actions import handle_invalid_action

scene = get_planner_scene()

handle_invalid_action(
    scene=scene,
    object_name="red block",
    instruction="pick up the red block"
)