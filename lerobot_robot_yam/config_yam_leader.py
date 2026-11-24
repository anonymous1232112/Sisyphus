from dataclasses import dataclass

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("yam_leader")
@dataclass
class YamLeaderConfig(TeleoperatorConfig):

    port: int = 5001
    host: str = "localhost"

    gripper: bool = True
