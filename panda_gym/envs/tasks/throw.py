from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class Throw(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type="sparse",
        distance_threshold: float = 0.01,
        goal_xy_range: float = 0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, 0])
        # self.goal_range_low = np.array([-goal_xy_range / 2, 0, 0])
        # self.goal_range_high = np.array([goal_xy_range / 2, 0, 0])
        self.fingers_indices = np.array([9, 10])
        self.joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10])
        self.ee_link = 11
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=0.0)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.5,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            lateral_friction=1.0,
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        goal = np.array([1.5, 0.0, self.object_size / 2])  # z offset for the cube center
        noise = np.random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        ee_pos = self.sim.get_link_position("panda",self.ee_link)
        joint_angles = np.zeros(7)
        for i in self.joint_indices[:7]:
            joint_angles[i] = self.sim.get_joint_angle("panda",i)
        joint_f_angles = np.append(joint_angles,[0.021, 0.021])
        self.sim.control_joints("panda",self.joint_indices,joint_f_angles,np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]))
        object_position = ee_pos
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def computed_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info) -> np.ndarray:
        if achieved_goal.shape[0] == 3:
            d = distance(achieved_goal, desired_goal)
            if d < self.distance_threshold:
                return np.array([100.0])
            return -d.astype(np.float32)
        elif achieved_goal.shape[1] == 3:
            d = distance(achieved_goal, desired_goal)
            for i in range(d.shape[0]):
                if d[i] < self.distance_threshold:
                    d[i] = 100.0
            return -d.astype(np.float32)
    
    def is_truncated(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, obj_vel: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if np.all(achieved_goal[3:6]<1e-4) and achieved_goal[2] - desired_goal[2] < 1e-3 and d < self.distance_threshold:
            return np.array(True)
