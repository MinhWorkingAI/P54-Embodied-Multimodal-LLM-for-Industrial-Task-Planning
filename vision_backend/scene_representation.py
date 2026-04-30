import json
from typing import Dict, Any, List


def calculate_center(bounding_box: List[int]) -> List[int]:
    """
    Calculate the center point of a bounding box.

    Bounding box format:
    [x, y, width, height]
    """
    x, y, width, height = bounding_box
    center_x = x + width // 2
    center_y = y + height // 2
    return [center_x, center_y]


def build_scene_representation(vision_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert object detection output into a structured scene representation.
    """
    scene = {}

    for obj in vision_output.get("objects", []):
        label = obj["label"]
        bounding_box = obj["bounding_box"]
        center = calculate_center(bounding_box)

        scene[label] = {
            "center": center,
            "bounding_box": bounding_box,
            "confidence": obj.get("confidence", None)
        }

    return {
        "image_id": vision_output.get("image_id", "unknown"),
        "scene": scene
    }


def load_json(filename: str) -> Dict[str, Any]:
    """
    Load JSON data from file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filename: str) -> None:
    """
    Save JSON data to file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_scene_map(filename: str = "scene_representation.json") -> Dict[str, Any]:
    """
    Load and return the structured scene map.
    This will be used by the task planner module.
    """
    return load_json(filename)


def main() -> None:
    vision_output = load_json("scene_output.json")
    scene_representation = build_scene_representation(vision_output)

    print(json.dumps(scene_representation, indent=2))
    save_json(scene_representation, "scene_representation.json")
    print("\nSaved to scene_representation.json")


if __name__ == "__main__":
    main()