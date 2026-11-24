import logging

import numpy as np

logger = logging.getLogger(__name__)


class YamRpcClient:
    def __init__(self, host: str, port: int, name: str):
        self.host = host
        self.port = port
        self.name = name
        self._client = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    def connect(self) -> None:
        try:
            import portal
        except ImportError as e:
            raise ImportError("`portal` is required: pip install 'portal>=3.7'") from e
        addr = f"{self.host}:{self.port}"
        logger.info("[%s] connecting to yam-server at %s", self.name, addr)
        self._client = portal.Client(addr)

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None

    def _require(self):
        if self._client is None:
            raise RuntimeError(f"[{self.name}] not connected to yam-server")
        return self._client

    def num_dofs(self) -> int:
        return int(self._require().num_dofs().result())

    def get_observations(self) -> dict[str, np.ndarray]:
        return self._require().get_observations().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._require().command_joint_pos(np.asarray(joint_pos, dtype=np.float64))
