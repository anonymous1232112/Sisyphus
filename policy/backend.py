from contextlib import ExitStack
import gc
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

from policy.common import task_environment


def seed_rng(seed, torch, numpy):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'[policy-server] task RNG seed={seed} (python/numpy/torch/cuda)', flush=True)


def trim_heads(policy, torch):
    removed = []
    for name, module in list(policy.named_modules()):
        if isinstance(module, torch.nn.Linear) and name.endswith('lm_head'):
            parent, attr = name.rsplit('.', 1)
            setattr(policy.get_submodule(parent), attr, torch.nn.Identity())
            removed.append(name)
            print('[lowmem] removed ' + name, flush=True)
    torch.cuda.empty_cache()
    return removed


def fingerprint(model):
    return tuple((name, (Path(model) / name).stat().st_size,
                  (Path(model) / name).stat().st_mtime_ns)
                 for name in ('config.json', 'model.safetensors'))


class ResidentBackend:
    def __init__(self, model):
        self.model = str(Path(model).resolve())
        self.policy = None
        self.loaded_signature = None

    def initialize(self):
        from policy.preflight import inspect_model
        report = inspect_model(Path(self.model))
        if report['status'] != 'metadata_checks_passed':
            raise ValueError(str(report['issues_requiring_review']))
        import numpy as np
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.pi05 import modeling_pi05 as M
        from lerobot.rollout import context
        from lerobot.utils.import_utils import register_third_party_plugins
        from lerobot.utils.process import ensure_multiprocessing_start_method
        from lerobot.utils.utils import init_logging
        from policy.noise import install_pi05_noise_scale
        init_logging()
        register_third_party_plugins()
        ensure_multiprocessing_start_method('spawn')
        self.torch, self.np, self.M = torch, np, M
        seed_rng(int(os.environ.get('PI05_SEED', '0')), torch, np)
        install_pi05_noise_scale(M)
        pc = PreTrainedConfig.from_pretrained(self.model, local_files_only=True)
        pc.pretrained_path = self.model
        pc.compile_model = False
        if pc.use_peft:
            raise ValueError('Resident service requires a complete pretrained_model, not a PEFT adapter')
        reports = []
        original = M.PI05Policy.load_state_dict

        def checked(instance, *args, **kwargs):
            result = original(instance, *args, **kwargs)
            reports.append(result)
            return result

        start = time.monotonic()
        signature = fingerprint(self.model)
        with patch.object(M.PI05Policy, 'load_state_dict', checked):
            policy = context._load_pretrained_policy(pc)
        if not reports or any(r.missing_keys or r.unexpected_keys for r in reports):
            raise RuntimeError('PI05 checkpoint did not load completely; refusing to serve')
        if fingerprint(self.model) != signature:
            raise RuntimeError('Model files changed while loading')
        if len(trim_heads(policy, torch)) != 2:
            raise RuntimeError('Expected the two validated PI05 lm_head layers')
        self.policy = policy.to('cuda').eval()
        self.loaded_signature = signature
        print(f'[policy-server] MODEL_LOADED_ONCE seconds={time.monotonic()-start:.3f} '
              f'GPU_GiB={torch.cuda.memory_allocated()/2**30:.2f}', flush=True)

    def parse_config(self, task):
        import draccus
        from lerobot.configs import parser
        from lerobot.rollout.configs import RolloutConfig
        with patch.object(sys, 'argv', ['lerobot-rollout', *task.argv]):
            args = parser.filter_path_args(RolloutConfig.__get_path_fields__(), task.argv)
            return draccus.parse(config_class=RolloutConfig, args=args)

    def run(self, task, control):
        from lerobot.rollout import context
        from lerobot.rollout.strategies import episodic
        from policy.noise import install_pi05_noise_scale

        if fingerprint(self.model) != self.loaded_signature:
            raise RuntimeError('Loaded model files changed; restart this policy service explicitly')
        robots, teleops, datasets, engines = [], [], [], []
        ctx = None
        cleanup_errors = []
        result = None

        def tracked(factory, objects, guard_actions=False):
            def make(*args, **kwargs):
                control.check()
                obj = factory(*args, **kwargs)
                objects.append(obj)
                if guard_actions:
                    send = obj.send_action

                    def guarded(action):
                        control.check()
                        return send(action)
                    obj.send_action = guarded
                return obj
            return make

        def cached_policy(config):
            if str(Path(config.pretrained_path).resolve()) != self.model:
                raise ValueError('Model mismatch before hardware connection')
            return self.policy

        def attempt(label, fn):
            try:
                fn()
            except Exception as exc:
                cleanup_errors.append(f'{label}: {type(exc).__name__}: {exc}')

        with task_environment(task):
            seed_rng(task.seed, self.torch, self.np)
            install_pi05_noise_scale(self.M)
            self.policy.reset()
            cfg = self.parse_config(task)
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(context, '_load_pretrained_policy', cached_policy))
                    stack.enter_context(patch.object(context, 'make_robot_from_config',
                        tracked(context.make_robot_from_config, robots, True)))
                    stack.enter_context(patch.object(context, 'make_teleoperator_from_config',
                        tracked(context.make_teleoperator_from_config, teleops)))
                    stack.enter_context(patch.object(context.LeRobotDataset, 'create',
                        tracked(context.LeRobotDataset.create, datasets)))
                    stack.enter_context(patch.object(context, 'create_inference_engine',
                        tracked(context.create_inference_engine, engines)))
                    stack.enter_context(patch.object(episodic, 'init_keyboard_listener',
                        lambda: (None, control.events)))
                    control.check()
                    ctx = context.build_rollout_context(cfg, control.cancelled)
                    control.check()
                    class ServerEpisode(episodic.EpisodicStrategy):
                        def _policy_loop(self, *args, **kwargs):
                            try:
                                return super()._policy_loop(*args, **kwargs)
                            finally:
                                self._engine.pause()

                    strategy = ServerEpisode(cfg.strategy)
                    strategy.setup(ctx)
                    strategy.run(ctx)
                    if ctx.policy.inference.failed:
                        raise RuntimeError('RTC inference thread failed')
            except Exception:
                if not control.reason:
                    raise
            finally:
                for engine in engines:
                    thread = getattr(engine, '_rtc_thread', None)
                    attempt('pause inference', engine.pause)
                    attempt('stop inference', engine.stop)
                    if thread is not None and thread.is_alive():
                        thread.join(timeout=5)
                        if thread.is_alive():
                            cleanup_errors.append('RTC thread still alive: model cannot be reused')
                for dataset in datasets:
                    attempt('finalize dataset', dataset.finalize)
                for robot in robots:
                    if hasattr(robot, 'arm'):
                        for name, camera in robot.cameras.items():
                            attempt('disconnect camera ' + name,
                                    lambda c=camera: c.disconnect() if c.is_connected else None)
                        attempt('disconnect follower RPC', robot.arm.disconnect)
                    else:
                        attempt('disconnect robot', robot.disconnect)
                for teleop in teleops:
                    attempt('disconnect teleop', teleop.disconnect)
                if cleanup_errors:
                    raise RuntimeError('; '.join(cleanup_errors))
            dataset = ctx.data.dataset if ctx is not None else None
            if not control.cancelled.is_set() and (dataset is None or dataset.num_episodes != 1
                                                   or dataset.num_frames == 0):
                raise RuntimeError('No complete nonempty episode was recorded')
            result = {'dataset_root': str(task.root),
                      'repo_id': dataset.repo_id if dataset is not None else None,
                      'episodes': dataset.num_episodes if dataset is not None else 0,
                      'frames': dataset.num_frames if dataset is not None else 0}
        gc.collect()
        return result
