import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot

from .config_yam_follower import YamFollowerConfig
from .rpc import YamRpcClient

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 6


def motor_names(gripper: bool) -> list[str]:
    names = [f"joint_{i}.pos" for i in range(NUM_ARM_JOINTS)]
    if gripper:
        names.append("gripper.pos")
    return names


class YamFollower(Robot):

    config_class = YamFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YamFollowerConfig):
        super().__init__(config)
        self.config = config
        self.arm = YamRpcClient(config.host, config.port, "yam_follower")
        self.cameras = make_cameras_from_configs(config.cameras)
        self._names = motor_names(config.gripper)
        self._soft_start_done = not config.soft_start

    @property
    def _motors_ft(self) -> dict[str, type]:
        return dict.fromkeys(self._names, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected and all(c.is_connected for c in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} already connected")
        self.arm.connect()
        n = self.arm.num_dofs()
        expected = len(self._names)
        if n != expected:
            self.arm.disconnect()
            raise RuntimeError(
                f"yam-server follower reports {n} DOF but this config expects {expected} "
                f"(--robot.gripper={self.config.gripper}). Check --follower-gripper of yam-server."
            )
        for cam in self.cameras.values():
            cam.connect()
        self._soft_start_done = not self.config.soft_start
        logger.info("yam_follower connected (%d DOF, %d cameras)", n, len(self.cameras))

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        self.arm.disconnect()
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        logger.info("%s disconnected", self.name)

    def _read_positions(self) -> np.ndarray:
        obs = self.arm.get_observations()
        q = np.asarray(obs["joint_pos"], dtype=np.float64)[:NUM_ARM_JOINTS]
        if self.config.gripper:
            g = np.asarray(obs.get("gripper_pos", [0.0]), dtype=np.float64).reshape(-1)[:1]
            q = np.concatenate([q, g])
        return q

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        q = self._read_positions()
        obs: dict[str, Any] = {k: float(v) for k, v in zip(self._names, q, strict=True)}
        for key, cam in self.cameras.items():
            t0 = time.perf_counter()
            obs[key] = cam.async_read()
            logger.debug("%s read %s: %.1f ms", self, key, (time.perf_counter() - t0) * 1e3)
        return obs

    def _step_limits(self) -> np.ndarray | None:
        limits = []
        if not self._soft_start_done:
            limits.append(np.full(len(self._names), self.config.soft_start_max_step))
        mrt = self.config.max_relative_target
        if mrt is not None:
            if isinstance(mrt, dict):
                limits.append(np.array([mrt.get(k, np.inf) for k in self._names], dtype=np.float64))
            else:
                limits.append(np.full(len(self._names), float(mrt)))
        return np.minimum.reduce(limits) if limits else None

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        goal = np.array([float(action[k]) for k in self._names], dtype=np.float64)

        limit = self._step_limits()
        if limit is not None:
            present = self._read_positions()
            delta = goal - present
            if not self._soft_start_done and np.all(np.abs(delta) < self.config.soft_start_tolerance):
                self._soft_start_done = True
                logger.info("yam_follower soft start complete, now tracking the leader")
                limit = self._step_limits()
            if limit is not None:
                goal = present + np.clip(delta, -limit, limit)

        self.arm.command_joint_pos(goal)
        return {k: float(v) for k, v in zip(self._names, goal, strict=True)}
