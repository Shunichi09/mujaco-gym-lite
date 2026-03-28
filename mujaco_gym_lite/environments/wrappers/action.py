import copy
from typing import Any, cast

import gymnasium
import numpy as np
import numpy.typing as npt
from gymnasium.core import ActionWrapper, Wrapper

from mujaco_gym_lite.environments.mujoco_env.drawer.drawer import DrawerRobotEnv
from mujaco_gym_lite.logger import logger


class FixedRotationActionWrapper(ActionWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        fixed_rotation: npt.NDArray,
        robot_rotation_key_name: str = "robot/end_effector/rotation",
    ):
        super().__init__(env)
        self._robot_rotation_key_name = robot_rotation_key_name
        self.action_space = self._build_action_space()
        self._fixed_rotation = fixed_rotation
        assert len(self._fixed_rotation) == 4

    def _build_action_space(self) -> gymnasium.spaces.Dict:
        assert isinstance(self.env.unwrapped, DrawerRobotEnv)
        original_action_space = cast(gymnasium.spaces.Dict, self.env.action_space)

        new_action_space = {}
        for space_name, space in original_action_space.items():
            if space_name == self._robot_rotation_key_name:
                continue
            logger.info(f"new action space: {space_name}, space info {space} at FixedRotationActionWrapper")
            new_action_space[space_name] = space

        return gymnasium.spaces.Dict(new_action_space)

    def action(self, action):
        action[self._robot_rotation_key_name] = self._fixed_rotation
        return action


class RemoveKeyActionWrapper(ActionWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        task_related_action_keys=["task/end_episode", "robot/home"],
    ):
        super().__init__(env)
        self._task_related_action_keys = task_related_action_keys
        self._removed_task_related_action_keys: list[str] = []
        self.action_space = self._build_action_space()

    def _build_action_space(self) -> gymnasium.spaces.Dict:
        assert isinstance(self.env.unwrapped, DrawerRobotEnv)
        original_action_space = cast(gymnasium.spaces.Dict, self.env.action_space)

        new_action_space = {}
        for space_name, space in original_action_space.items():
            if space_name in self._task_related_action_keys:
                self._removed_task_related_action_keys.append(space_name)
                continue
            logger.info(f"new action space: {space_name}, space info {space} at RemoveKeyActionWrapper")
            new_action_space[space_name] = space

        return gymnasium.spaces.Dict(new_action_space)

    def action(self, action):
        for removed_task_related_action_key in self._removed_task_related_action_keys:
            if removed_task_related_action_key == "task/end_episode":
                action[removed_task_related_action_key] = np.zeros(1, dtype=np.float32)
            elif removed_task_related_action_key == "robot/home":
                action[removed_task_related_action_key] = np.zeros(1, dtype=np.float32)
            else:
                raise NotImplementedError

        return action


class DiscreteToAbsolutePositionActionWrapper(Wrapper):
    def __init__(
        self, env: gymnasium.Env, absolute_positions: list[npt.NDArray], action_key: str = "robot/end_effector/position"
    ):
        super().__init__(env)
        self._absolute_positions = absolute_positions
        self._num_actions = len(self._absolute_positions)
        self._action_key = action_key
        self.action_space = self._build_action_space()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        state, state_info = self.env.reset(seed=seed, options=options)
        assert isinstance(state, dict)
        return state, state_info

    def step(self, action):
        discrete_action = action["robot/end_effector/position"]
        discrete_action = (
            discrete_action.flatten()[0] if isinstance(discrete_action, np.ndarray) else int(discrete_action)
        )
        action["robot/end_effector/position"] = self._absolute_positions[int(discrete_action)]
        next_state, reward, termination, truncated, info = self.env.step(action)
        return next_state, reward, termination, truncated, info

    def _build_action_space(self) -> gymnasium.spaces.Dict:
        assert isinstance(self.env.unwrapped, DrawerRobotEnv)
        original_action_space = self.env.action_space
        original_action_space = cast(gymnasium.spaces.Dict, self.env.action_space)
        new_action_space = copy.deepcopy(original_action_space)
        assert "robot/end_effector/position" in list(original_action_space.keys())
        new_action_space["robot/end_effector/position"] = gymnasium.spaces.Discrete(n=self._num_actions)
        return new_action_space


class LinearInterpolationPositionActionWrapper(Wrapper):
    def __init__(self, env: gymnasium.Env, max_position_action: float = 0.02):
        super().__init__(env)
        assert max_position_action > 0, "max_position_action must be > 0"
        self._max_position_action = max_position_action

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        state, state_info = self.env.reset(seed=seed, options=options)
        if not isinstance(state, dict) or "robot/end_effector/position" not in state:
            raise ValueError("State must be a dict containing key 'robot/end_effector/position'")
        self._current_robot_position = np.array(state["robot/end_effector/position"])
        return state, state_info

    def step(self, action: dict) -> tuple[Any, float, bool, bool, dict]:
        target_position = np.array(action["robot/end_effector/position"])
        diff = target_position - self._current_robot_position
        distance = np.linalg.norm(diff)

        num_steps = max(1, int(np.ceil(distance / self._max_position_action)))

        qpos_internals = []
        qvel_internals = []
        for step in range(1, num_steps + 1):
            intermediate_position = self._current_robot_position + diff * (step / num_steps)
            action["robot/end_effector/position"] = intermediate_position
            next_state, reward, termination, truncated, step_info = self.env.step(action)

            qpos_internals.append(next_state["qpos"])
            qvel_internals.append(next_state["qvel"])
            self.render()
            if termination or truncated:
                break

        self._current_robot_position = np.array(next_state["robot/end_effector/position"], dtype=np.float32)
        step_info["internal_qpos"] = np.array(qpos_internals, dtype=np.float32)
        step_info["internal_qvel"] = np.array(qvel_internals, dtype=np.float32)

        return next_state, reward, termination, truncated, step_info


class FixedCameraActionWrapper(ActionWrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        camera_position: npt.NDArray,
        camera_rotation: npt.NDArray,
        camera_position_key_name: str = "camera/mocap_camera/position",
        camera_rotation_key_name: str = "camera/mocap_camera/rotation",
    ):
        super().__init__(env)
        self._camera_position = camera_position
        self._camera_rotation = camera_rotation
        self._camera_position_key_name = camera_position_key_name
        self._camera_rotation_key_name = camera_rotation_key_name
        self.action_space = self._build_action_space()

    def _build_action_space(self) -> gymnasium.spaces.Dict:
        assert isinstance(self.env.unwrapped, DrawerRobotEnv)
        original_action_space = cast(gymnasium.spaces.Dict, self.env.action_space)

        new_action_space = {}
        for space_name, space in original_action_space.items():
            if space_name == self._camera_position_key_name or space_name == self._camera_rotation_key_name:
                continue
            logger.info(f"new action space: {space_name}, space info {space} at FixedCameraActionWrapper")
            new_action_space[space_name] = space

        return gymnasium.spaces.Dict(new_action_space)

    def action(self, action):
        action[self._camera_position_key_name] = self._camera_position
        action[self._camera_rotation_key_name] = self._camera_rotation
        return action
