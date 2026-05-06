from vision_backend.scene_representation import get_planner_scene
from vision_backend.safety_checks import validate_object_exists


scene = get_planner_scene()

print("Scene loaded:")
print(scene)

validate_object_exists(scene, "red block")
print("red block exists ")

validate_object_exists(scene, "yellow block")
print("yellow block exists ")