from collections import deque
import math

import numpy as np


def validate_early_stop(value):
    if value not in ('0', '1'):
        raise ValueError('SIS_EARLY_STOP must be 0 or 1')
    return value == '1'


def measured_state(observation):
    if 'observation.state' in observation:
        state = observation['observation.state']
    else:
        state = [observation[f'joint_{i}.pos'] for i in range(6)] + [observation['gripper.pos']]
    state = np.asarray(state, dtype=float)
    if state.shape != (7,) or not np.isfinite(state).all():
        raise ValueError('Early-stop requires seven finite measured state values')
    return state


class EarlyStopDetector:
    def __init__(self, enabled=True, fk=None, release_fn=None, home_state=None):
        if fk is None:
            from bon.core import YamFK
            fk = YamFK()
        self.fk, self.enabled, self.release_fn = fk, bool(enabled), release_fn
        self.home_tip = None if home_state is None else self.fk.tip(np.asarray(home_state, float)[:6])
        self.window = deque(maxlen=15)
        self.first_state = self.first_tip = None
        self.closed_seen = self.opened_after_close = False
        self.close_run = self.open_run = self.samples = 0
        self.closed_at_s = self.opened_at_s = None
        self.last = None
        self.last_elapsed = -1.

    def update(self, state, elapsed_s):
        state = measured_state({'observation.state': state})
        elapsed = float(elapsed_s)
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed < self.last_elapsed:
            raise ValueError('Elapsed observation time must be finite, nonnegative and monotonic')
        self.last_elapsed = elapsed
        if self.first_state is None:
            self.first_state = state.copy()
            self.first_tip = self.fk.tip(state[:6]) if self.home_tip is None else self.home_tip
        self.samples += 1
        self.window.append(state[:6].copy())
        g = float(state[6])
        self.close_run = self.close_run + 1 if g < .5 else 0
        if not callable(self.release_fn):
            raise ValueError('release_fn (released after an Eq.8 hold) is required')
        if not self.closed_seen and self.release_fn():
            self.closed_seen = True
            self.closed_at_s = elapsed
        self.open_run = self.open_run + 1 if self.closed_seen and g > .8 else 0
        if self.open_run >= 10 and not self.opened_after_close:
            self.opened_after_close = True
            self.opened_at_s = elapsed
        tip = self.fk.tip(state[:6])
        distance = float(np.linalg.norm(tip - self.first_tip))
        variation = float(np.ptp(np.asarray(self.window), axis=0).max())
        conditions = dict(close_then_open=self.closed_seen and self.opened_after_close,
                          home_after_open=self.opened_after_close and distance < .06,
                          stationary=len(self.window) == 15 and variation < .01,
                          minimum_duration=elapsed >= 3.)
        self.last = dict(elapsed_s=elapsed, samples=self.samples, gripper_measured=g,
            close_run=self.close_run, open_run=self.open_run, closed_seen=self.closed_seen,
            opened_after_close=self.opened_after_close, closed_at_s=self.closed_at_s,
            opened_at_s=self.opened_at_s, tip_distance_m=distance,tip_xyz_now=tip.tolist(),
            joint_window_samples=len(self.window), max_joint_range_rad=variation,
            conditions=conditions,
            thresholds=dict(closed_g_lt=.5, open_g_gt=.8,
                consecutive_steps=10, home_distance_lt_m=.06, stationary_steps=15,
                joint_range_lt_rad=.01, minimum_elapsed_s=3.),
            first_state=self.first_state.tolist(), end_state=state.tolist())
        self.last['would_stop'] = all(conditions.values())
        self.last['early_stop_enabled'] = self.enabled
        return self.enabled and self.last['would_stop']
