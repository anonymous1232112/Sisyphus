from contextlib import ExitStack
import threading
import time
import json
from unittest.mock import patch

from policy.backend import ResidentBackend
from policy.server import Control, Service, TaskCancelled


class AtomicControl(Control):
    def __init__(self):
        super().__init__()
        self.target_gate = threading.RLock()
        self.task_failed = False

    def cancel(self, reason, code=130):
        with self.target_gate:
            super().cancel(reason, code)

    def send_target(self, send, *args, **kwargs):
        with self.target_gate:
            self.check()
            if self.task_failed:
                raise TaskCancelled(self.reason or 'Task failed: new targets disabled')
            return send(*args, **kwargs)

    def fail_task(self,reason,code=1):
        with self.target_gate:
            with self.lock:
                if not self.cancelled.is_set():
                    self.reason=str(reason)
                    self.code=code
                self.task_failed=True
                self.events['stop_recording']=True


class ResidentService(Service):
    def key(self, key):
        if key in ('q', 'esc') and self.active is not None:
            self.active.cancel(key + ' in resident BoN service terminal', 130)
            print('\n[resident-bon] q: target submission disabled; saving episode.', flush=True)
        elif key == 'n' and self.active is not None:
            self.active.events['exit_early'] = True


def guard_target_factory(factory, control, observation_sink=None, action_sink=None, smoother=None):
    def make(*args, **kwargs):
        robot = factory(*args, **kwargs)
        arm = getattr(robot, 'arm', None)
        if arm is None or not hasattr(arm, 'command_joint_pos'):
            raise TypeError('Resident BoN requires the verified YAM RPC adapter')
        send = arm.command_joint_pos
        def send_and_note(*a, **k):
            result = send(*a, **k)
            if action_sink is not None:
                target = a[0] if a else k['joint_pos']
                action_sink(target)
            values = a[0] if a else k['joint_pos']
            robot._resident3_last_action = dict(zip(
                [*[f'joint_{i}.pos' for i in range(6)],'gripper.pos'],map(float,values)))
            return result
        arm.command_joint_pos = lambda *a, **k: control.send_target(send_and_note, *a, **k)
        if smoother is not None:
            names=[*[f'joint_{i}.pos' for i in range(6)],'gripper.pos']
            send_action=robot.send_action
            def smooth_action(action):
                control.check()
                values=smoother.apply([action[k] for k in names])
                return send_action(dict(zip(names,values)))
            robot.send_action=smooth_action
        if observation_sink is not None:
            get_observation = robot.get_observation
            def timed_observation(*a, **k):
                began = time.perf_counter()
                obs = get_observation(*a, **k)
                observation_sink(time.perf_counter()-began)
                return obs
            robot.get_observation = timed_observation
        return robot
    return make


class SharedResidentBackend(ResidentBackend):
    def initialize(self):
        super().initialize()
        from service.warmup import warmup_model
        self.warmup_report = warmup_model(self.policy, self.model)
        print('[resident-bon3] BON_WARMUP_COMPLETE ' + json.dumps(self.warmup_report), flush=True)

    def parse_config(self, task):
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot_robot_yam.config_yam_follower import YamFollowerConfig
        from lerobot_robot_yam.config_yam_leader import YamLeaderConfig
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        cfg = super().parse_config(task)
        ds = cfg.dataset
        if not (ds.streaming_encoding and ds.encoder_threads == 2 and
                ds.encoder_queue_maxsize == 8 and ds.num_image_writer_processes == 0 and
                ds.num_image_writer_threads_per_camera == 0):
            raise ValueError('Bounded streaming recording settings required')
        return cfg

    def run(self, task, control):
        from lerobot.rollout import context
        from lerobot.rollout.strategies import episodic
        from service.earlystop import EarlyStopDetector
        from .loop import policy_loop
        from service.engine import write_log
        from service.smoothing import SmoothSettings,ActionSmoother
        from .detector import WrongGraspDetector,scene_from_target
        from bon.core import Settings
        detector = EarlyStopDetector(enabled=task.early_stop == '1',
                                     home_state=task.abort_settings['home_state'] if task.continuation else None)
        target_name,scene_points,missing=scene_from_target(task.env.get('SIS_TARGET_JSON'))
        abort_detector=WrongGraspDetector(target_name,scene_points,settings=task.abort_settings['holding'],
            fk=detector.fk,missing=missing,abort_action_enabled=task.abort_action_enabled)
        detector.release_fn = abort_detector.released_after_holding
        h_report = dict(h_rule_version=task.h_rule_version, relocate_on_empty=task.relocate_on_empty,
                        continuation=task.continuation)
        end_records = []
        smooth_settings=SmoothSettings.parse(task.action_smooth) if task.action_smooth is not None else SmoothSettings()
        smoother=ActionSmoother(smooth_settings) if smooth_settings.mode!='0' else None
        def finish(record):
            record.update(abort_detector.report())
            record.update(h_report)
            record.update(dataset=task.dataset, inference='bon' if int(task.env.get('SIS_BON') or '0') > 1 else 'rtc')
            record.update(action_smoothing=smooth_settings.report(),
                          dataset_action='submitted_filtered_then_limited_target' if smoother else 'original_policy_action')
            if record['wrong_grasp_abort']:
                record['dataset_action']='policy: same original filter/limiter; recovery: guarded RPC, OpenPlan + go_home quintic limits'
            end_records.append(record)
            write_log(task.env.get('SIS_BON_LOG') or task.root / 'bon.jsonl', record)
            (task.root / 'execution.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
            print('[resident-bon3] EXECUTION_END ' + json.dumps(record, ensure_ascii=False), flush=True)
        def record_grasp(record):
            write_log(task.env.get('SIS_BON_LOG') or task.root/'bon.jsonl',
                      dict(record,dataset=task.dataset,wall_time_unix_s=time.time()))
        original_strategy = episodic.EpisodicStrategy
        class MeasuredStopEpisode(original_strategy):
            def _policy_loop(self, *args, **kwargs):
                return policy_loop(self, *args, **kwargs, detector=detector,
                                   control=control, finish=finish, stop_on_empty=task.relocate_on_empty,
                                   abort_detector=abort_detector,record_grasp=record_grasp,
                                   abort_home_state=task.abort_settings['home_state'])
        n = int(task.env.get('SIS_BON') or '0')
        settings = Settings.from_env(task.env, mode='batch') if n > 1 else None
        if settings is not None and settings.fallback_reason:
            raise ValueError('BoN target/calibration invalid before device connection: ' + settings.fallback_reason)
        if settings is not None:
            settings.holding_fn = abort_detector.holding
        previous_config = self.policy.config.rtc_config
        previous_policy_processor = getattr(self.policy, 'rtc_processor', None)
        previous_model_processor = getattr(self.policy.model, 'rtc_processor', None)
        original_factory = context.create_inference_engine
        datasets = []
        bon_engines = []
        original_dataset_create = context.LeRobotDataset.create

        def dataset_create(*args, **kwargs):
            dataset = original_dataset_create(*args, **kwargs)
            datasets.append(dataset)
            encoder = dataset.writer._streaming_encoder
            if dataset.writer.image_writer is not None or encoder is None or encoder.queue_maxsize != 8:
                raise RuntimeError('Dataset did not create the required bounded streaming writer')
            return dataset

        def factory(config, **kwargs):
            if n <= 1:
                return original_factory(config, **kwargs)
            from service.engine import AsyncBonEngine, SyncSharedBonEngine
            self.policy.config.rtc_config = None
            self.policy.rtc_processor = None
            self.policy.model.rtc_processor = None
            cls = AsyncBonEngine if task.bon_engine == 'async_shared' else SyncSharedBonEngine
            engine = cls(settings, control.cancelled.is_set, exec_steps=task.exec_steps,
                         blend_steps=task.blend_steps, **kwargs)
            bon_engines.append(engine)
            return engine

        print(f'[resident-bon] task n={n} engine={task.bon_engine if n > 1 else "ordinary_rtc"} '
              f'exec_steps={task.exec_steps if n > 1 else "RTC unchanged"} '
              f'blend_steps={task.blend_steps if n > 1 else "RTC unchanged"} '
              f'early_stop={task.early_stop} action_smoothing={smooth_settings.report()} '
              'streaming=true encoder_threads=2 queue_per_camera=8', flush=True)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(episodic, 'EpisodicStrategy', MeasuredStopEpisode))
                def observation_sink(elapsed):
                    for engine in bon_engines:
                        engine.note_observation(elapsed)
                def action_sink(target):
                    for engine in bon_engines:
                        engine.note_action_sent(target)
                stack.enter_context(patch.object(context, 'make_robot_from_config',
                    guard_target_factory(context.make_robot_from_config, control,
                                         observation_sink if n > 1 else None,
                                         action_sink if n > 1 else None,smoother=smoother)))
                stack.enter_context(patch.object(context, 'create_inference_engine', factory))
                stack.enter_context(patch.object(context.LeRobotDataset, 'create', dataset_create))
                result = super().run(task, control)
                dropped = {str(i): dict(ds.writer._streaming_encoder._dropped_frames)
                           for i, ds in enumerate(datasets)}
                if any(any(count > 0 for count in counts.values()) for counts in dropped.values()):
                    raise RuntimeError('Streaming encoder dropped frames; task not reported as successful: ' + str(dropped))
                result.update(streaming_encoding=True, encoder_threads=2, encoder_queue_maxsize=8,
                              dropped_video_frames=dropped, bon_engine=task.bon_engine if n > 1 else 'ordinary_rtc')
                if end_records:
                    result.update(stop_reason=end_records[-1]['stop_reason'],
                                  end_time_s=end_records[-1]['end_time_s'])
                    result.update(abort_detector.report(),abort_recovery=end_records[-1].get('abort_recovery'))
                    result.update(h_report)
                    result.update({key:end_records[-1].get(key) for key in
                        ('recovery_ok','settle_error_rad','tip_dist_m','recovery_failure_kind')})
                    result['fatal']=False
                return result
        finally:
            if not any(e._rtc_thread is not None and e._rtc_thread.is_alive() for e in bon_engines):
                self.policy.config.rtc_config = previous_config
                self.policy.rtc_processor = previous_policy_processor
                self.policy.model.rtc_processor = previous_model_processor
