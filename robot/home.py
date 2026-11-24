from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time

JOINT_NAMES = [f'joint_{i}.pos' for i in range(6)]
STATE_NAMES = JOINT_NAMES + ['gripper.pos']


def finite_vector(values, lengths=(6, 7)):
    try:
        result = tuple(float(x) for x in values)
    except (TypeError, ValueError) as exc:
        raise ValueError('Expected a finite joint state') from exc
    if len(result) not in lengths or not all(math.isfinite(x) for x in result):
        raise ValueError(f'Expected {lengths} finite state values')
    return result


def dataset_states(root, episode_index=0):
    import pyarrow.parquet as pq
    root = Path(root).expanduser().resolve()
    info = json.loads((root / 'meta/info.json').read_text())
    feature = info['features']['observation.state']
    if feature.get('names') != STATE_NAMES or feature.get('shape') != [7]:
        raise ValueError('Dataset state ordering must be joint_0..5.pos, gripper.pos')
    first = last = None
    seen = set()
    for path in sorted((root / 'data').rglob('*.parquet')):
        table = pq.read_table(path, columns=['episode_index', 'frame_index', 'observation.state'])
        for row in table.to_pylist():
            if row['episode_index'] != episode_index:
                continue
            index = int(row['frame_index'])
            if index in seen:
                raise ValueError('Duplicate frame_index in the selected episode')
            seen.add(index)
            state = finite_vector(row['observation.state'], (7,))
            if first is None or index < first[0]: first = (index, state)
            if last is None or index > last[0]: last = (index, state)
    if first is None or first[0] != 0:
        raise ValueError('Selected rollout episode has no recorded frame zero')
    return first[1], last[1]


@dataclass(frozen=True)
class HomePose:
    state: tuple
    source: str

    @property
    def joints(self):
        return self.state[:6]


def load_home(*, home_reference=None, home_dataset=None, episode_index=0):
    if (home_reference is None) == (home_dataset is None):
        raise ValueError('Choose exactly one home_reference or home_dataset')
    if home_dataset is not None:
        root = Path(home_dataset).expanduser().resolve()
        first, _ = dataset_states(root, episode_index)
        return HomePose(first, f'{root} / episode {episode_index}, frame 0')
    path = Path(home_reference).expanduser().resolve()
    if path.is_dir(): path /= 'log.jsonl'
    references = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip(): continue
        try: row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f'Malformed home-reference log at line {number}') from exc
        if row.get('event') == 'home_reference':
            references.append(finite_vector(row.get('state')))
    if not references:
        raise ValueError('No home_reference event in ' + str(path))
    if any(row[:6] != references[0][:6] for row in references[1:]):
        raise ValueError('Conflicting home_reference events; select an unambiguous source')
    return HomePose(references[0], str(path) + ' / home_reference')


@dataclass(frozen=True)
class MotionLimits:
    fps: float = 30.
    max_step: float = .02
    max_velocity: float = .15
    max_acceleration: float = .3
    min_seconds: float = 3.
    tracking_limit: float = .15
    tolerance: float = .02
    settle_seconds: float = 5.
    rpc_timeout: float = 2.

    def validate(self):
        for value in vars(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('All motion limits must be finite and positive')
        if not 1 <= self.fps <= 30 or self.max_step > .02:
            raise ValueError('Rate must be 1..30 Hz; max-step must be <=0.02 rad')
        if self.max_velocity > .6 or self.max_acceleration > 1.:
            raise ValueError('Velocity <=0.6 rad/s and acceleration <=1 rad/s^2 required')
        if self.tolerance > .05 or self.tracking_limit > .3 or self.tracking_limit < self.max_step:
            raise ValueError('Tolerance <=0.05 and tracking limit in [max-step,0.3] required')
        return self


class QuinticPath:
    def __init__(self, start, home, limits):
        self.start = finite_vector(start, (6,))
        self.home = finite_vector(home, (6,))
        self.delta = tuple(b-a for a, b in zip(self.start, self.home))
        limits.validate()
        distance = max(map(abs, self.delta))
        velocity = min(limits.max_velocity, limits.max_step*limits.fps)
        duration = max(limits.min_seconds, 1.875*distance/velocity,
                       math.sqrt((10/math.sqrt(3))*distance/limits.max_acceleration)) * 1.01
        self.duration = math.ceil(duration*limits.fps)/limits.fps

    def at(self, elapsed):
        u = max(0., min(1., elapsed/self.duration))
        s = u*u*u*(10 + u*(-15 + 6*u))
        return tuple(a+s*d for a, d in zip(self.start, self.delta))


class MotionStopped(Exception):
    pass


class StopGate:
    def __init__(self):
        self.event = threading.Event()
        self.lock = threading.RLock()

    def stop(self):
        with self.lock: self.event.set()

    def check(self):
        if self.event.is_set(): raise MotionStopped('q / interruption')

    def submit(self, send, values):
        with self.lock:
            self.check()
            return send(values)


def wait_result(future, gate, timeout):
    deadline = time.monotonic() + timeout
    while True:
        gate.check()
        remaining = deadline-time.monotonic()
        if remaining <= 0: raise TimeoutError('Follower RPC timed out; no further target will be sent')
        try: return future.result(timeout=min(.01, remaining))
        except TimeoutError: continue


def drive_home(rpc, home, initial, limits, gate, *, clock=time.monotonic,
               wait=None, output=print):
    limits.validate()
    wait = wait or gate.event.wait
    initial = finite_vector(initial, (7,))
    current = finite_vector(rpc.read(), (7,))
    gate.check()
    if max(abs(a-b) for a, b in zip(current, initial)) > .02:
        raise RuntimeError('Pose changed during confirmation; rerun and confirm the new pose')
    path = QuinticPath(current[:6], home.joints, limits)
    grip = current[6]
    last = current[:6]
    began = previous_tick = clock()
    last_progress = -1.
    sent = 0
    while True:
        gate.check()
        remaining = 1/limits.fps-(clock()-previous_tick)
        if remaining > 0: wait(remaining)
        gate.check()
        current = finite_vector(rpc.read(), (7,))
        gate.check()
        if max(abs(a-b) for a, b in zip(current[:6], last)) > limits.tracking_limit:
            raise RuntimeError('Follower tracking error/external movement exceeds limit; stopping')
        now = clock()
        elapsed = now-began
        target = path.at(elapsed)
        if max(abs(a-b) for a, b in zip(target, last)) > limits.max_step + 1e-10:
            raise RuntimeError('Control delay would exceed max-step; stopping without a catch-up target')
        rpc.send(target+(grip,))
        last, previous_tick = target, now
        sent += 1
        if elapsed-last_progress >= 1.:
            output(f'[go_home] {elapsed:.1f}/{path.duration:.1f}s '
                   f'max measured home error={max(abs(a-b) for a,b in zip(current[:6],home.joints)):.4f} rad')
            last_progress = elapsed
        if elapsed >= path.duration: break
    settled, deadline = 0, clock()+limits.settle_seconds
    while clock() < deadline:
        gate.check()
        current = finite_vector(rpc.read(), (7,))
        gate.check()
        error = max(abs(a-b) for a,b in zip(current[:6],home.joints))
        settled = settled+1 if error <= limits.tolerance else 0
        if settled >= 3:
            return dict(stop_reason='home_reached', elapsed_s=clock()-began,
                        targets_sent=sent, max_joint_error_rad=error,
                        gripper_target=grip, source=home.source)
        wait(1/limits.fps)
    raise RuntimeError('Home settling timed out; no further targets will be sent')
