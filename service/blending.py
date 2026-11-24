from dataclasses import dataclass
import math

import numpy as np


def validate_blend_steps(value):
    if type(value) is int:
        result = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        result = int(value)
    else:
        raise ValueError('SIS_BON_BLEND_STEPS must be an integer in 0..50')
    if not 0 <= result <= 50:
        raise ValueError('SIS_BON_BLEND_STEPS must be an integer in 0..50')
    return result


@dataclass
class PreviousPlan:
    actions: np.ndarray
    first_index: int
    committed_at: float
    consumed: int = 0


def blend_segment(segment, *, steps, last_sent, previous_plan=None, now=0.,
                  fps=30., waited=False, initial_state=None):
    steps = validate_blend_steps(steps)
    raw = np.asarray(segment)
    if raw.ndim != 2 or raw.shape[1] != 7 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Expected a finite nonempty [steps,7] execution segment')
    if not np.issubdtype(raw.dtype, np.floating):
        raw = raw.astype(float)
    output = raw.copy()
    has_previous = last_sent is not None
    baseline = np.asarray(last_sent if has_previous else initial_state, dtype=float)
    if baseline.shape != (7,) or not np.isfinite(baseline).all():
        raise ValueError('Expected finite last target / initial measured state')
    k = min(steps, len(raw)) if has_previous else 0
    old_rows = np.repeat(baseline[None, :], k, axis=0)
    old_count, old_start = 0, None
    if k and previous_plan is not None and not waited:
        elapsed_slots = max(0, math.floor((now-previous_plan.committed_at)*fps+1e-7))
        old_start = previous_plan.first_index + max(previous_plan.consumed, elapsed_slots)
        old_count = min(k, max(0, len(previous_plan.actions)-old_start))
        if old_count:
            old_rows[:old_count] = previous_plan.actions[old_start:old_start+old_count]
    if k:
        weights = (np.arange(1, k+1, dtype=float)/(k+1))[:, None]
        output[:k, :6] = (1-weights)*old_rows[:, :6] + weights*raw[:k, :6]
    before = float(np.max(np.abs(raw[0, :6]-baseline[:6])))
    after = float(np.max(np.abs(output[0, :6]-baseline[:6])))
    window = min(len(raw), k+1) if k else 1
    raw_chain = np.vstack((baseline[:6], raw[:window, :6]))
    blend_chain = np.vstack((baseline[:6], output[:window, :6]))
    rec = dict(blend_steps=k, blend_steps_requested=steps,
        joint_jump_before_rad=before, joint_jump_after_rad=after,
        blend_window_max_jump_before_rad=float(np.abs(np.diff(raw_chain,axis=0)).max()),
        blend_window_max_jump_after_rad=float(np.abs(np.diff(blend_chain,axis=0)).max()),
        blend_old_plan_steps=old_count, blend_last_sent_steps=k-old_count,
        blend_old_start_index_zero_based=old_start,
        blend_reference=('initial_no_previous_action' if not has_previous else
                         'disabled' if not k else 'last_sent_after_underrun' if waited else
                         'old_plan' if old_count==k else 'old_plan_and_last_sent' if old_count else 'last_sent'),
        joint_jump_reference='last_submitted_target' if has_previous else 'initial_measured_state')
    return output, rec
