from gymnasium.envs.registration import register

register(
    id="FourRooms-v0",
    entry_point="mujaco_gym_lite.environments.tabular.four_rooms.four_rooms_env:FourRooms",
    max_episode_steps=1000,
)

register(
    id="ObserverFourRooms-v0",
    entry_point="mujaco_gym_lite.environments.tabular.observer_four_rooms.observer_four_rooms_env:ObserverFourRooms",
    max_episode_steps=1000,
)


register(
    id="DrawerRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.drawer.drawer:DrawerRobotEnv",
    max_episode_steps=500,
)

register(
    id="MugRackRobot-v0",
    entry_point="mujaco_gym_lite.environments.mujoco_env.mug_rack.mug_rack:MugRackRobotEnv",
    max_episode_steps=1000,
)
