from typing import Union

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.environment_tools.mujoco_env.env_models.env_model import EnvModel
from mujaco_gym_lite.environment_tools.mujoco_env.functions.contact import get_contact_info_between_abstract_geom_names
from mujaco_gym_lite.environment_tools.mujoco_env.functions.mj_data import site_pose
from mujaco_gym_lite.utils.transforms import create_transformation_matrix, extract_position, extract_rotation


class SoccerBall(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        ball_joint_name: str,
        ball_body_names: list[str],
        geom_root_name: str,
        ball_geom_name: str,
        ball_max_position: npt.NDArray,
        ball_min_position: npt.NDArray,
        ball_radius: float = 0.026,
    ):
        super().__init__(base_body_name, [ball_joint_name], ball_body_names, geom_root_name)
        self._ball_geom_name = ball_geom_name
        self._ball_radius = ball_radius
        self._ball_max_position = ball_max_position
        self._ball_min_position = ball_min_position

    def has_ball_touch(
        self, abstract_geom_name: str, exclude_geom_names: list[str] = []
    ) -> tuple[bool, dict[str, Union[list[npt.NDArray], list[tuple[str, str]]]]]:
        num_contact, contact_names, contact_points = get_contact_info_between_abstract_geom_names(
            mj_model=self._mj_model,
            mj_data=self._mj_data,
            geom1_abstract_name=self._ball_geom_name,
            geom2_abstract_name=abstract_geom_name,
            exclude_abstract_names=exclude_geom_names,
        )
        return bool(num_contact > 0), {
            "ball/contact_positions": contact_points,
            "ball/contact_names": contact_names,
        }


class SoccerGoal(EnvModel):
    def __init__(
        self,
        base_body_name: str,
        goal_body_names: list[str],
        geom_root_name: str,
        goal_left_front_site_name: str,
        goal_right_front_site_name: str,
        goal_left_back_site_name: str,
        goal_right_back_site_name: str,
        goal_center_top_site_name: str,
        goal_center_bottom_back_site_name: str,
    ):
        super().__init__(base_body_name, [], goal_body_names, geom_root_name)
        self._goal_left_front_site_name = goal_left_front_site_name
        self._goal_right_front_site_name = goal_right_front_site_name
        self._goal_left_back_site_name = goal_left_back_site_name
        self._goal_right_back_site_name = goal_right_back_site_name
        self._goal_center_top_site_name = goal_center_top_site_name
        self._goal_center_bottom_back_site_name = goal_center_bottom_back_site_name

    def is_ball_in_goal(self, ball: SoccerBall) -> bool:
        ball_position = extract_position(ball.body_pose()[0])
        goal_position = extract_position(self.body_pose()[0])
        goal_rotation = extract_rotation(self.body_pose()[0], rotation_type="matrix")

        w_T_ball = create_transformation_matrix(ball_position)
        w_T_goal = create_transformation_matrix(goal_position, goal_rotation)
        w_T_goal_front_left = site_pose(self._mj_data, [self._goal_left_front_site_name])[0]
        w_T_goal_front_right = site_pose(self._mj_data, [self._goal_right_front_site_name])[0]
        w_T_goal_back_left = site_pose(self._mj_data, [self._goal_left_back_site_name])[0]
        w_T_goal_back_right = site_pose(self._mj_data, [self._goal_right_back_site_name])[0]
        w_T_goal_center_top = site_pose(self._mj_data, [self._goal_center_top_site_name])[0]

        goal_T_w = np.linalg.inv(w_T_goal)
        goal_T_ball = extract_position(goal_T_w @ w_T_ball)
        goal_T_left_front = extract_position(goal_T_w @ w_T_goal_front_left)
        goal_T_right_front = extract_position(goal_T_w @ w_T_goal_front_right)
        goal_T_left_back = extract_position(goal_T_w @ w_T_goal_back_left)
        goal_T_right_back = extract_position(goal_T_w @ w_T_goal_back_right)
        goal_T_center_top = extract_position(goal_T_w @ w_T_goal_center_top)
        ball_radius = ball._ball_radius
        goal_height = goal_T_center_top[2]

        goal_corners = np.stack([goal_T_left_front, goal_T_right_front, goal_T_left_back, goal_T_right_back], axis=0)
        up_axis = 2
        a1, a2 = [i for i in range(3) if i != up_axis]

        a1_min, a1_max = goal_corners[:, a1].min(), goal_corners[:, a1].max()
        a2_min, a2_max = goal_corners[:, a2].min(), goal_corners[:, a2].max()

        floor_up = goal_corners[:, up_axis].min()
        goal_height = goal_T_center_top[up_axis] - floor_up
        inside_a1 = (a1_min + ball_radius) <= goal_T_ball[a1] <= (a1_max - ball_radius)
        inside_a2 = (a2_min + ball_radius) <= goal_T_ball[a2] <= a2_max
        inside_up = (floor_up + ball_radius) <= goal_T_ball[up_axis] <= (floor_up + goal_height - ball_radius)

        return bool(inside_a1 and inside_a2 and inside_up)

    def center_goal_position(self) -> npt.NDArray:
        center_position = extract_position(site_pose(self._mj_data, [self._goal_center_bottom_back_site_name])[0])
        return center_position
