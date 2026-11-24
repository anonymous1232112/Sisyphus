from dataclasses import dataclass, field
import time


@dataclass
class SampleResult:
    candidates: object
    timing: dict = field(default_factory=dict)


class TimedSampler:
    def __init__(self, policy, pre, post, *, device, task, robot_type, n=5,
                 minimum_latency_s=0., clock=time.perf_counter):
        self.policy, self.pre, self.post = policy, pre, post
        self.device, self.task, self.robot_type, self.n = device, task, robot_type, n
        self.minimum_latency_s, self.clock = minimum_latency_s, clock

    def __call__(self, obs, cancelled=lambda: False, *, noise=None):
        import torch
        from contextlib import nullcontext
        from lerobot.policies.utils import prepare_observation_for_inference
        from bon.prefix import generate_shared
        device = torch.device(self.device or 'cpu')
        gpu = device.type == 'cuda'
        def sync():
            if gpu:
                torch.cuda.synchronize(device)
        began = self.clock()
        timing = {}
        amp = torch.autocast(device_type='cuda') if gpu and self.policy.config.use_amp else nullcontext()
        with torch.inference_mode(), amp:
            prepared = prepare_observation_for_inference(dict(obs), device, self.task, self.robot_type)
            batch = self.pre(prepared)
            sync()
            inference_start = self.clock()
            timing['preprocess_s'] = inference_start - began
            events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if gpu else None
            if events:
                events[0].record()
            post_started = [None]
            def post(raw):
                if events:
                    events[1].record()
                sync()
                post_started[0] = self.clock()
                timing['gpu_inference_wall_s'] = post_started[0] - inference_start
                timing['gpu_inference_s'] = events[0].elapsed_time(events[1]) / 1000 if events else None
                return self.post(raw)
            diagnostics = {}
            candidate = generate_shared(self.policy, batch, post, self.n, 'batch',
                noise=noise, cancelled=cancelled, diagnostics=diagnostics)
            finished = self.clock()
            if candidate is not None and self.n > 1 and (
                    diagnostics.get('prefix_calls') != 1 or diagnostics.get('prefix_observation_batch') != 1):
                raise RuntimeError('Expected one shared prefix per observation')
            timing['postprocess_s'] = finished - post_started[0] if post_started[0] is not None else None
            timing['sampler_compute_s'] = finished - began
        delay_start = self.clock()
        while candidate is not None and not cancelled() and self.clock() - began < self.minimum_latency_s:
            time.sleep(min(.005, max(0., self.minimum_latency_s - (self.clock() - began))))
        timing['simulated_delay_s'] = self.clock() - delay_start if self.minimum_latency_s else 0.
        timing['cuda_events_used'] = gpu
        return SampleResult(None if cancelled() else candidate, timing)
