import pybullet as p
import pybullet_data
import numpy as np
import time
from simulation_backend.mock_robot import CommandResult

class RealRobot:

    def __init__(self, robot_id, object_registry):

        self.robot_id = robot_id
        self.object_registry = object_registry

        # Change this depending on robot type
        # KUKA iiwa usually uses 6
        self.end_effector_index = 6

        self.current_constraint = None
        self.held_object = None

        self.workspace_limits = {
            "x": [0.1, 0.8],
            "y": [-0.4, 0.4],
            "z": [0.0, 0.6]
        }

        self.home_position = [0.4, 0.0, 0.4]

        self.num_joints = p.getNumJoints(self.robot_id)

        print(f"[INFO] RealRobot initialized")
        print(f"[INFO] Robot joints: {self.num_joints}")

    def load_scene(self, scene):

        self.scene = scene

        print("[INFO] Scene loaded into robot")

    # ============================================================
    # UTILITY FUNCTIONS
    # ============================================================

    def step(self, steps=240):

        for _ in range(steps):
            p.stepSimulation()
            time.sleep(1 / 240)

    def validate_workspace(self, position):

        x, y, z = position

        if not (self.workspace_limits["x"][0] <= x <= self.workspace_limits["x"][1]):
            raise ValueError(f"X position out of bounds: {x}")

        if not (self.workspace_limits["y"][0] <= y <= self.workspace_limits["y"][1]):
            raise ValueError(f"Y position out of bounds: {y}")

        if not (self.workspace_limits["z"][0] <= z <= self.workspace_limits["z"][1]):
            raise ValueError(f"Z position out of bounds: {z}")

    def get_end_effector_position(self):

        state = p.getLinkState(self.robot_id, self.end_effector_index)

        return state[0]

    # ============================================================
    # ROBOT MOVEMENT
    # ============================================================

    def move_to(self, target_position, steps=50):

        self.validate_workspace(target_position)

        current_position = self.get_end_effector_position()

        path = np.linspace(current_position, target_position, steps)

        for point in path:

            joint_positions = p.calculateInverseKinematics(
                bodyUniqueId=self.robot_id,
                endEffectorLinkIndex=self.end_effector_index,
                targetPosition=point
            )

            for joint_index in range(self.num_joints):

                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[joint_index],
                    force=500
                )

            p.stepSimulation()
            time.sleep(1 / 240)
       
        return CommandResult(
            success=True,
            command="move",
            message=f"Moved to {target_position}"
        )
    # ============================================================
    # GRIPPER FUNCTIONS
    # ============================================================

    def open_gripper(self):

        print("[INFO] Gripper opened")

    def close_gripper(self):

        print("[INFO] Gripper closed")

    # ============================================================
    # OBJECT FUNCTIONS
    # ============================================================

    def locate(self, object_name):

        entry = self.object_registry.get_by_label(object_name)

        if entry is None:
            return CommandResult(
                success=False,
                command="locate",
                message=f"Object not found: {object_name}"
            )

        return CommandResult(
            success=True,
            command="locate",
            message=f"Found '{object_name}' at {entry.position}"
        )

    def attach_object(self, object_id):

        if self.current_constraint is not None:
            return

        constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.end_effector_index,

            childBodyUniqueId=object_id,
            childLinkIndex=-1,

            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],

            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )

        self.current_constraint = constraint_id

    def release_object(self):

        if self.current_constraint is not None:

            p.removeConstraint(self.current_constraint)

            self.current_constraint = None
            self.held_object = None

    # ============================================================
    # PICK FUNCTION
    # ============================================================

    def pick(self, object_name):

        print(f"[INFO] Picking object: {object_name}")

        entry = self.object_registry.get_by_label(object_name)

        if entry is None:
            raise ValueError(f"Object not found: {object_name}")

        object_id = entry.body_id

        object_position, _ = p.getBasePositionAndOrientation(object_id)

        x, y, z = object_position

        hover_position = [x, y, z + 0.15]
        grasp_position = [x, y, z + 0.03]

        self.move_to(hover_position)

        self.move_to(grasp_position)

        self.close_gripper()

        self.attach_object(object_id)

        self.held_object = object_name

        self.move_to(hover_position)

        print(f"[SUCCESS] Picked {object_name}")

        return CommandResult(
            success=True,
            command="pick",
            message=f"Picked {object_name}"
        )

    def move_to_object(self, object_name):

        entry = self.object_registry.get_by_label(object_name)

        if entry is None:
            return CommandResult(
                success=False,
                command="move",
                message=f"Object not found: {object_name}"
            )

        x, y, z = entry.position

        hover_position = [x, y, z + 0.15]

        self.move_to(hover_position)

        return CommandResult(
            success=True,
            command="move",
            message=f"Moved above {object_name}"
        )
    # ============================================================
    # PLACE FUNCTION
    # ============================================================

    def place(self, target_position):

        print(f"[INFO] Placing object at: {target_position}")

        if self.held_object is None:
            raise ValueError("No object currently held")

        x, y, z = target_position

        hover_position = [x, y, z + 0.15]

        place_position = [x, y, z + 0.03]

        # Move above target
        self.move_to(hover_position)

        # Move down
        self.move_to(place_position)

        # Release object
        self.open_gripper()

        self.release_object()

        # Move back up
        self.move_to(hover_position)

        print("[SUCCESS] Object placed")
        
        return CommandResult(
            success=True,
            command="place",
            message="Object placed"
        )
    # ============================================================
    # RESET FUNCTION
    # ============================================================

    def reset(self):

        print("[INFO] Resetting robot")

        self.release_object()

        self.move_to(self.home_position)

    # ============================================================
    # EXECUTE COMMANDS
    # ============================================================

    def execute_command(self, command):

        action = command.get("action")

        if action == "move":

            target = command.get("target")

            self.move_to(target)

        elif action == "pick":

            object_name = command.get("object")

            self.pick(object_name)

        elif action == "place":

            target = command.get("target")

            self.place(target)

        elif action == "locate":

            object_name = command.get("object")

            return self.locate(object_name)

        else:
            raise ValueError(f"Unknown action: {action}")