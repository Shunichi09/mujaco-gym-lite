import os
import pathlib
import sys
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from mujaco_gym_lite.utils.math import fit_angle_in_range
from mujaco_gym_lite.utils.transforms import matrix_to_quat

sys.path.append(str(pathlib.Path(os.path.abspath(__file__)).parent / "ikfast_binds"))  # noqa

import j2n6s300_ikfast  # noqa


class InverseKinematicsSolver(metaclass=ABCMeta):
    def __init__(self, core_lib):
        self._core_lib = core_lib

    def compute_fk(self, joint_angles: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        actual_ee_pos, actual_ee_mat = self._core_lib.compute_fk(joint_values=joint_angles)
        actual_ee_quat = matrix_to_quat(np.array(actual_ee_mat).reshape(3, 3))
        return actual_ee_pos, actual_ee_quat

    def solve_ik(
        self,
        target_position: npt.NDArray,
        target_rotation: npt.NDArray,
        current_joint_angles: npt.NDArray,
        free_joint_values: List[float] = [],
    ) -> Optional[npt.NDArray]:
        solutions = self._core_lib.solve_ik(
            target_position=target_position,
            target_rotation=target_rotation.flatten(),
            free_joint_values=free_joint_values,
        )
        solutions = np.array(solutions)
        if len(solutions) == 0:
            # logger.info("No inverse kinematics solusions.")
            return None
        else:
            solutions = self._fit_angle_in_range(solutions, np.array(current_joint_angles).flatten())
            return self.sort_solution(solutions, np.array(current_joint_angles).flatten())

    def sort_solution(self, solutions: npt.NDArray, current_joint_angles: npt.NDArray) -> npt.NDArray:
        assert len(solutions) > 0
        assert solutions.shape[1] == len(current_joint_angles)
        diff = np.linalg.norm(
            fit_angle_in_range(solutions - current_joint_angles),
            axis=1,
        )
        sort_idx = np.argsort(diff)
        return solutions[sort_idx]

    @abstractmethod
    def _fit_angle_in_range(self, solutions, current_joint_angles):
        raise NotImplementedError


class J2n6s300IKSolver(InverseKinematicsSolver):
    def __init__(self, core_lib=j2n6s300_ikfast):
        super().__init__(core_lib)

    def _fit_angle_in_range(self, solutions, current_joint_angles):
        fitted_solutions = []
        for sol in solutions:
            fitted_sol = [
                self._fit_angle(sol[0], current_joint_angles[0]),
                self._fit_angle(sol[1], current_joint_angles[1]),
                self._fit_angle(sol[2], current_joint_angles[2]),
                self._fit_angle(sol[3], current_joint_angles[3]),
                self._fit_angle(sol[4], current_joint_angles[4]),
                self._fit_angle(sol[5], current_joint_angles[5]),
            ]
            fitted_solutions.append(np.array(fitted_sol).flatten())
        return np.array(fitted_solutions)

    def _fit_angle(self, solution_angle: float, current_angle: float):
        raw_diff = solution_angle - current_angle
        positive_one_rotation_diff = solution_angle + 2.0 * np.pi - current_angle
        negatice_one_rotation_diff = solution_angle - 2.0 * np.pi - current_angle
        diffs = np.abs([raw_diff, positive_one_rotation_diff, negatice_one_rotation_diff])
        if np.argmin(diffs) == 0:
            return solution_angle
        elif np.argmin(diffs) == 1:
            return solution_angle + 2.0 * np.pi
        elif np.argmin(diffs) == 2:
            return solution_angle - 2.0 * np.pi
