from gymnasium.envs.registration import register

register(
    id="DrawerRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.drawer.drawer:DrawerRobotEnv",
    max_episode_steps=None,
)

register(
    id="ButtonRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.button.button:ButtonRobotEnv",
    max_episode_steps=None,
)

register(
    id="WindowRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.window.window:WindowRobotEnv",
    max_episode_steps=None,
)

register(
    id="SlidingDoorRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.door.door:SlidingDoorRobotEnv",
    max_episode_steps=None,
)

register(
    id="AssemblyRingRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.assembly_ring.assembly_ring:AssemblyRingRobotEnv",
    max_episode_steps=None,
)

register(
    id="RingHangingRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.ring.ring:RingHangingRobotEnv",
    max_episode_steps=None,
)

register(
    id="HingedBoxRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.hinged_box.hinged_box:HingedBoxRobotEnv",
    max_episode_steps=None,
)

register(
    id="LeverRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.lever.lever:LeverRobotEnv",
    max_episode_steps=None,
)

register(
    id="SoccerRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.soccer.soccer:SoccerRobotEnv",
    max_episode_steps=None,
)

register(
    id="LiddedBoxRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.lidded_box.lidded_box:LiddedBoxRobotEnv",
    max_episode_steps=None,
)
