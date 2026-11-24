from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
ENV_KEYS = ('PI05_NOISE_SCALE', 'SIS_BON', 'SIS_TARGET_JSON', 'SIS_CALIB_JSON', 'SIS_BON_LOG')
MAX_PACKET = 128 * 1024


def address(value):
    value = value.removeprefix('tcp://')
    host, port = value.rsplit(':', 1)
    if host not in ('127.0.0.1', 'localhost') or not 1 <= int(port) <= 65535:
        raise ValueError('Use tcp://127.0.0.1:PORT (local connections only)')
    return '127.0.0.1', int(port)


def rollout_args(model, dataset, instruction, seconds):
    return [
        '--strategy.type=episodic', f'--policy.path={model}',
        '--inference.type=rtc', '--inference.rtc.execution_horizon=10',
        '--inference.rtc.max_guidance_weight=10.0', '--device=cuda',
        '--return_to_initial_position=false', '--robot.type=yam_follower',
        '--robot.max_relative_target=0.07', '--robot.soft_start=false',
        "--robot.cameras={top: {type: intelrealsense, serial_number_or_name: '260322272085', width: 640, height: 480, fps: 30}, wrist: {type: intelrealsense, serial_number_or_name: '260422275259', width: 640, height: 480, fps: 30}}",
        '--teleop.type=yam_leader', f'--dataset.repo_id=local/{dataset}',
        f'--dataset.root=./data/{dataset}', f'--dataset.single_task={instruction}',
        '--dataset.fps=30', '--dataset.num_episodes=1',
        f'--dataset.episode_time_s={seconds}', '--dataset.push_to_hub=false',
        '--display_data=false',
    ]


@dataclass
class Task:
    model: str
    dataset: str
    instruction: str
    seed: int
    seconds: str
    env: dict
    argv: list
    timeout_s: float

    @classmethod
    def parse(cls, obj):
        if not isinstance(obj, dict) or set(obj) != {
            'version', 'model', 'dataset', 'instruction', 'seed', 'seconds', 'env', 'argv', 'timeout_s'
        } or obj['version'] != 1:
            raise ValueError('Invalid task protocol/schema')
        model, dataset, instruction = obj['model'], obj['dataset'], obj['instruction']
        if not isinstance(model, str) or not Path(model).is_absolute():
            raise ValueError('MODEL must be an absolute local directory')
        model = str(Path(model).resolve())
        if not isinstance(dataset, str) or not re.fullmatch(r'rollout_[A-Za-z0-9_.-]+', dataset):
            raise ValueError('Dataset name must start with rollout_ and contain no slashes')
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 8192:
            raise ValueError('Invalid instruction')
        if type(obj['seed']) is not int or not 0 <= obj['seed'] <= 4294967295:
            raise ValueError('Seed must be an integer in [0, 4294967295]')
        seconds = obj['seconds']
        if not isinstance(seconds, str) or not re.fullmatch(r'[0-9]+(?:[.][0-9]+)?', seconds):
            raise ValueError('Invalid seconds')
        if not 0 < float(seconds) <= 600:
            raise ValueError('Service episodes must be between 0 and 600 seconds')
        timeout = float(obj['timeout_s'])
        if not math.isfinite(timeout) or not 1 <= timeout <= 3600:
            raise ValueError('Task timeout must be finite and in [1, 3600] seconds')
        env = obj['env']
        if not isinstance(env, dict) or set(env) != set(ENV_KEYS):
            raise ValueError('Only the five permitted task environment variables may be sent')
        if any(v is not None and (not isinstance(v, str) or len(v) > 8192 or '\0' in v)
               for v in env.values()):
            raise ValueError('Invalid task environment value')
        scale = float(env['PI05_NOISE_SCALE'] if env['PI05_NOISE_SCALE'] is not None else '1.0')
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('Invalid PI05_NOISE_SCALE')
        bon = env['SIS_BON'] or '0'
        if not bon.isdigit() or int(bon) > 1:
            raise ValueError('B3 is paused: SIS_BON > 1 is not implemented; refusing before hardware access')
        if obj['argv'] != rollout_args(model, dataset, instruction, seconds):
            raise ValueError('Rollout arguments differ from the verified run_once settings')
        return cls(model, dataset, instruction, obj['seed'], seconds, dict(env), obj['argv'], timeout)

    @property
    def root(self):
        return ROOT / 'data' / self.dataset


def packet(sock, obj):
    data = (json.dumps(obj, ensure_ascii=False) + '\n').encode()
    if len(data) > MAX_PACKET:
        raise ValueError('Packet too large')
    sock.sendall(data)


def receive(stream):
    raw = stream.readline(MAX_PACKET + 1)
    if not raw or len(raw) > MAX_PACKET or not raw.endswith(b'\n'):
        raise ValueError('Connection closed or invalid packet size/framing')
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError('Expected a JSON object')
    return obj


@contextmanager
def task_environment(task):
    values = {**task.env, 'PI05_SEED': str(task.seed)}
    previous = {k: os.environ.get(k) for k in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
