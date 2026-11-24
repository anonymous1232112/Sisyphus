from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

import numpy as np

from bon.core import make_selector
from bon.selector import endpoints
from .execution import execution_window, choose_exec_steps, percentile90, validate_exec_steps
from .sampling import SampleResult, TimedSampler
from .blending import PreviousPlan, blend_segment, validate_blend_steps

NAMES = [f'joint_{i}.pos' for i in range(6)] + ['gripper.pos']


def write_log(path, record):
    if not path:
        raise ValueError('A BoN log path is required')
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


@dataclass
class InferenceResult:
    candidates: object
    observed: float
    latency: float
    ready: float
    timing: dict


def select_segment(selector, candidates, state, start, end):
    selector.h = end - 1
    rec = selector.choose(candidates, state)
    target = selector.t.get('object' if rec['stage'] == 1 else 'region')
    scores = None
    if target is not None:
        xy = endpoints(selector.fk, candidates, state, end - 1)[:, :2]
        scores = np.linalg.norm(xy - target[:2], axis=1).tolist()
    rec.update(scores=scores, search_enabled=rec['reason'] == 'searched',
               score_step_one_based=end, score_index_zero_based=end - 1,
               execution_slice_zero_based=[start, end],
               score_step_gripper_targets=candidates[:, end - 1, 6].tolist(),
               gripper_measured=float(state[6]))
    selector.log[:] = selector.log[-1:]
    return candidates[rec['index'], start:end].copy(), rec


class AsyncBonEngine:
    mode = 'async_shared'
    def __init__(self, settings, cancelled=lambda: False, *, sampler=None,
                 clock=time.monotonic, log_sink=None, action_wrap=None,
                 exec_steps='auto', blend_steps=5, **kw):
        if list(kw['dataset_features']['action']['names']) != NAMES or list(kw['ordered_action_keys']) != NAMES:
            raise ValueError('Verified YAM action ordering required')
        self.settings, self.cancelled = settings, cancelled
        self.policy, self.pre, self.post = kw['policy'], kw['preprocessor'], kw['postprocessor']
        self.fps, self.horizon = float(kw['fps']), 10
        self.exec_setting = validate_exec_steps(exec_steps)
        self.blend_steps = validate_blend_steps(blend_steps)
        self.latencies = deque(maxlen=5)
        self.global_shutdown = kw.get('shutdown_event')
        self.clock = clock
        self.selector = make_selector(settings)
        self.log_sink = log_sink or (lambda rec: write_log(settings.log_path, rec))
        self.sampler = sampler or self._sampler(kw)
        if action_wrap is None:
            import torch
            action_wrap = lambda row: torch.from_numpy(row.copy()).float()
        self.action_wrap = action_wrap
        self.condition = threading.Condition()
        self.active, self.stopping = threading.Event(), threading.Event()
        self._rtc_thread = None
        self.request = self.result = None
        self.inflight, self.error = False, None
        self.queue = deque(maxlen=50)
        self.chunk = self.underrun_ticks = 0
        self.current = None
        self.observation_timing = {'observation_s': None}
        self.underrun_started = None
        self.total_underrun_s, self.wait_count = 0., 0
        self.previous_plan = None
        self.last_sent = None
        self.latest_record = None

    def _sampler(self, kw):
        return TimedSampler(self.policy, self.pre, self.post, device=kw['device'],
            task=kw['task'], robot_type=kw['robot_wrapper'].robot_type, n=self.settings.n)

    def note_observation(self, elapsed_s):
        self.observation_timing = {'observation_s': float(elapsed_s)}

    def note_action_sent(self, target):
        target = np.asarray(target, dtype=float)
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError('Expected a finite submitted YAM target')
        self.last_sent = target.copy()

    @property
    def ready(self):
        return True

    @property
    def failed(self):
        return self.error is not None

    def _cancelled(self):
        return self.cancelled() or self.stopping.is_set() or not self.active.is_set() or (
            self.global_shutdown is not None and self.global_shutdown.is_set())

    def notify_observation(self, obs):
        pass

    def start(self):
        if self._rtc_thread is not None:
            raise RuntimeError('Engine already started')
        self._rtc_thread = threading.Thread(target=self._worker, name='SharedBoN', daemon=True)
        self._rtc_thread.start()

    def reset(self):
        self.active.clear()
        with self.condition:
            if self.inflight:
                raise RuntimeError('Cannot reset a live inference worker')
            self.request = self.result = None
            self.queue.clear()
            self.current = None
            self.error = None
            self.chunk = self.underrun_ticks = 0
            self.latencies.clear()
            self.underrun_started = None
            self.total_underrun_s, self.wait_count = 0., 0
            self.horizon = 10
            self.observation_timing = {'observation_s': None}
            self.previous_plan = None
            self.last_sent = None
            self.latest_record = None
        self.policy.reset(); self.pre.reset(); self.post.reset()
        self.selector = make_selector(self.settings)

    def resume(self):
        self.active.set()
        with self.condition:
            self.condition.notify_all()

    def _finish_segment(self, completed):
        if self.current is not None:
            self.log_sink(dict(event='chunk_end', chunk=self.current['chunk'],
                consumed_steps=self.current['consumed'], planned_steps=self.current['exec_steps'],
                completed=bool(completed),
                last_consumed_step_one_based=self.current['start'] + self.current['consumed']))
            self.current = None

    def pause(self):
        self.active.clear()
        self._finish_underrun(self.clock())
        with self.condition:
            self.request = self.result = None
            self.condition.notify_all()
        self._finish_segment(False)
        self.queue.clear()
        self.previous_plan = None

    def stop(self):
        self.pause()
        self.stopping.set()
        with self.condition:
            self.condition.notify_all()
        if self._rtc_thread is not None and self._rtc_thread.is_alive():
            self._rtc_thread.join(timeout=5)
            if self._rtc_thread.is_alive():
                raise RuntimeError('BoN worker still alive; resident model must not be reused')

    def _request(self, obs, now):
        with self.condition:
            if self.request is None and self.result is None and not self.inflight and not self._cancelled():
                copy_started = self.clock()
                snapshot = {k: (v.copy() if hasattr(v, 'copy') else
                                v.clone() if hasattr(v, 'clone') else v) for k, v in obs.items()}
                queued_at = self.clock()
                timing = dict(self.observation_timing, observation_copy_s=max(0., queued_at-copy_started),
                              observation_to_request_s=max(0., copy_started-now), _queued_at=queued_at)
                self.request = snapshot, now, timing
                self.condition.notify_all()

    def _worker(self):
        while not self.stopping.is_set():
            with self.condition:
                self.condition.wait_for(lambda: self.stopping.is_set() or (
                    self.active.is_set() and self.request is not None))
                if self.stopping.is_set():
                    return
                obs, observed, request_timing = self.request
                self.request, self.inflight = None, True
                self.condition.notify_all()
            start = self.clock()
            result = None
            try:
                sampled = self.sampler(obs, self._cancelled)
                timing = dict(request_timing)
                if isinstance(sampled, SampleResult):
                    candidates = sampled.candidates
                    timing.update(sampled.timing)
                else:
                    candidates = sampled
                if candidates is not None:
                    candidates = np.asarray(candidates)
                    if (candidates.ndim != 3 or candidates.shape[0] != self.settings.n or
                            candidates.shape[2] != 7 or not np.isfinite(candidates).all()):
                        raise ValueError('Invalid candidate array')
                if candidates is not None:
                    ready = self.clock()
                    timing['worker_queue_wait_s'] = max(0., start-timing.pop('_queued_at'))
                    compute = timing.get('sampler_compute_s')
                    timing['worker_other_s'] = (max(0.,ready-start-compute-timing.get('simulated_delay_s',0.))
                                                if compute is not None else None)
                    result = InferenceResult(candidates, observed, ready-start, ready, timing)
            except Exception as exc:
                if not self._cancelled():
                    self.error = f'{type(exc).__name__}: {exc}'
                    if self.global_shutdown is not None:
                        self.global_shutdown.set()
            finally:
                with self.condition:
                    if not self._cancelled():
                        self.result = result
                    self.inflight = False
                    self.condition.notify_all()
            if self.error:
                return

    def _adopt(self, result, obs, now):
        candidates, observed, latency, ready = result.candidates, result.observed, result.latency, result.ready
        age = max(0., now - observed)
        self.latencies.append(latency)
        self.horizon = choose_exec_steps(self.exec_setting, self.latencies, self.fps)
        start, end = execution_window(latency, self.fps, candidates.shape[1], self.horizon)
        state = np.asarray(obs['observation.state'], float).copy()
        score_start = time.perf_counter()
        chosen, rec = select_segment(self.selector, candidates, state, start, end)
        score_s = time.perf_counter()-score_start
        blend_start = time.perf_counter()
        chosen, blend_rec = blend_segment(chosen, steps=self.blend_steps,
            last_sent=self.last_sent, previous_plan=self.previous_plan,
            now=now, fps=self.fps, waited=self.underrun_started is not None,
            initial_state=state)
        self.previous_plan = PreviousPlan(candidates[rec['index']].copy(), start, now)
        timing = dict(result.timing, fk_score_s=score_s)
        timing['blend_s'] = time.perf_counter()-blend_start
        post = timing.get('postprocess_s')
        timing['denorm_and_fk_s'] = post+score_s if post is not None else None
        underrun_s = self._finish_underrun(now)
        self.queue.extend(chosen)
        rec.update(event='chunk', chunk=self.chunk, mode=self.mode, engine_version=3,
            tip_xyz_now=self.selector.fk.tip(state[:6]).tolist(),
            state_source='current_control_observation_at_adoption',
            n_requested=self.settings.n, n_generated=len(candidates),
            inference_latency_s=latency, observation_to_commit_s=age,
            elapsed_s=latency, delay_steps=start, d=start, exec_steps=self.horizon,
            exec_steps_setting=self.exec_setting, latency_p90_last5_s=percentile90(self.latencies),
            latency_history_s=list(self.latencies), ready_wait_s=max(0., now - ready),
            underrun_s=underrun_s, total_underrun_s=self.total_underrun_s, timing_s=timing,
            underrun_ticks=self.underrun_ticks, state=state.tolist(), fallback=None,
            generated_steps=candidates.shape[1], execution_steps=self.horizon)
        rec.update(blend_rec)
        self.latest_record = rec
        self.log_sink(rec)
        self.current = dict(chunk=self.chunk, start=start, consumed=0, exec_steps=self.horizon)
        self.chunk += 1
        self.underrun_ticks = 0

    def _finish_underrun(self, now):
        if self.underrun_started is None:
            return 0.
        elapsed = max(0., now-self.underrun_started)
        self.total_underrun_s += elapsed
        self.underrun_started = None
        return elapsed

    def wait_statistics(self):
        pending = max(0., self.clock()-self.underrun_started) if self.underrun_started is not None else 0.
        return dict(wait_count=self.wait_count, total_underrun_s=self.total_underrun_s+pending)

    def get_action(self, obs_frame):
        if self.error:
            raise RuntimeError('BoN worker failed: ' + self.error)
        if self._cancelled() or obs_frame is None:
            return None
        now = self.clock()
        if not self.queue:
            with self.condition:
                result, self.result = self.result, None
            if result is not None:
                self._adopt(result, obs_frame, now)
                self._request(obs_frame, now)
        if self.queue:
            row = self.queue.popleft()
            self.current['consumed'] += 1
            self.previous_plan.consumed += 1
            if not self.queue:
                self._finish_segment(True)
            return None if self._cancelled() else self.action_wrap(row)
        self.underrun_ticks += 1
        if self.underrun_started is None:
            self.underrun_started = now
            self.wait_count += 1
        self._request(obs_frame, now)
        return None


class SyncSharedBonEngine(AsyncBonEngine):
    mode = 'sync_shared'
    def start(self):
        pass

    def get_action(self, obs_frame):
        if self._cancelled() or obs_frame is None:
            return None
        if not self.queue:
            start = self.clock()
            sampled = self.sampler(dict(obs_frame), self._cancelled)
            candidates = sampled.candidates if isinstance(sampled, SampleResult) else sampled
            if candidates is None or self._cancelled():
                return None
            now = self.clock()
            timing = dict(self.observation_timing)
            if isinstance(sampled, SampleResult):
                timing.update(sampled.timing)
            self._adopt(InferenceResult(candidates, start, now-start, now, timing), obs_frame, now)
        row = self.queue.popleft()
        self.current['consumed'] += 1
        self.previous_plan.consumed += 1
        if not self.queue:
            self._finish_segment(True)
        return None if self._cancelled() else self.action_wrap(row)
