from typing import Any

import gymnasium


class QposQvelInfoWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        state, info = super().reset(seed=seed, options=options)
        info.update(
            {
                "qpos": self.env.unwrapped.data.qpos.copy(),
                "qvel": self.env.unwrapped.data.qvel.copy(),
            }
        )
        return state, info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = super().step(action)
        info.update(
            {
                "qpos": self.env.unwrapped.data.qpos.copy(),
                "qvel": self.env.unwrapped.data.qvel.copy(),
            }
        )
        return observation, reward, terminated, truncated, info
