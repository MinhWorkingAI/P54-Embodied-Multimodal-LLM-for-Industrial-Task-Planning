"""
simulation_backend
------------------
Execution-side package for the robotics pipeline.

Modules:
    action_schema   : RobotCommand / ActionPlan data contract
    mock_robot      : PyBullet-free robot simulator (drop-in for the real arm)
    executor        : Runs an ActionPlan on a robot instance step by step
    display_scene   : Scene visualisation helpers
"""
