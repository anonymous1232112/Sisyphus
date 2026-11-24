from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("yam_follower")
@dataclass
class YamFollowerConfig(RobotConfig):

    port: int = 1234
    host: str = "localhost"

    gripper: bool = True

    max_relative_target: float | dict[str, float] | None = None

    soft_start: bool = True
    soft_start_max_step: float = 0.02
    soft_start_tolerance: float = 0.05

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
