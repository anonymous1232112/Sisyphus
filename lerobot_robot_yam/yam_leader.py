import logging
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.teleoperators import Teleoperator

from .config_yam_leader import YamLeaderConfig
from .rpc import YamRpcClient
from .yam_follower import NUM_ARM_JOINTS, motor_names

logger = logging.getLogger(__name__)


class YamLeader(Teleoperator):

    config_class = YamLeaderConfig
    name = "yam_leader"

    def __init__(self, config: YamLeaderConfig):
        super().__init__(config)
        self.config = config
        self.arm = YamRpcClient(config.host, config.port, "yam_leader")
        self._names = motor_names(config.gripper)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._names, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} already connected")
        self.arm.connect()
        n = self.arm.num_dofs()
        if n < NUM_ARM_JOINTS:
            self.arm.disconnect()
            raise RuntimeError(f"yam-server leader reports {n} DOF, expected at least {NUM_ARM_JOINTS}")
        logger.info("%s connected", self.name)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        obs = self.arm.get_observations()
        q = np.asarray(obs["joint_pos"], dtype=np.float64)[:NUM_ARM_JOINTS]
        if self.config.gripper:
            g = float(np.asarray(obs.get("gripper_pos", [1.0])).reshape(-1)[0])
            q = np.concatenate([q, [np.clip(g, 0.0, 1.0)]])
        return {k: float(v) for k, v in zip(self._names, q, strict=True)}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        self.arm.disconnect()
        logger.info("%s disconnected", self.name)
