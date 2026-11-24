from dataclasses import asdict, dataclass
from policy.common import Task, rollout_args
from .execution import validate_exec_steps
from .blending import validate_blend_steps
from .smoothing import SmoothSettings

STREAM_ARGS = [
    '--dataset.streaming_encoding=true', '--dataset.encoder_threads=2',
    '--dataset.encoder_queue_maxsize=8', '--dataset.num_image_writer_processes=0',
    '--dataset.num_image_writer_threads_per_camera=0', '--play_sounds=false',
]


def resident_args(model, dataset, instruction, seconds):
    return rollout_args(model, dataset, instruction, seconds) + STREAM_ARGS


@dataclass
class ResidentTask(Task):
    bon_engine: str = 'async_shared'
    exec_steps: str = 'auto'
    blend_steps: int = 5
    early_stop: str = '1'
    action_smooth: dict | None = None

    @classmethod
    def parse(cls, obj):
        if not isinstance(obj, dict):
            raise ValueError('Expected task object')
        data = dict(obj)
        engine = data.pop('bon_engine', 'async_shared')
        exec_steps = data.pop('exec_steps', 'auto')
        blend_steps = data.pop('blend_steps', 5)
        early_stop = data.pop('early_stop', '1')
        smoothing = SmoothSettings.parse(data.pop('action_smooth',asdict(SmoothSettings())))
        if early_stop not in ('0', '1'):
            raise ValueError('SIS_EARLY_STOP must be 0 or 1')
        if engine not in ('async_shared', 'sync_shared'):
            raise ValueError('bon_engine must be async_shared or sync_shared')
        env = data.get('env')
        if not isinstance(env, dict):
            raise ValueError('Invalid task environment')
        value = env.get('SIS_BON') or '0'
        if not isinstance(value, str) or not value.isascii() or not value.isdigit() or int(value) > 16:
            raise ValueError('SIS_BON must be 0..16')
        exec_steps = validate_exec_steps(exec_steps) if int(value) > 1 else 'auto'
        blend_steps = validate_blend_steps(blend_steps) if int(value) > 1 else 0
        expected = resident_args(data['model'], data['dataset'], data['instruction'], data['seconds'])
        if data.get('argv') != expected:
            raise ValueError('Arguments differ from resident streaming settings')
        checked = Task.parse(dict(data, env={**env, 'SIS_BON': '0'},
            argv=rollout_args(data['model'], data['dataset'], data['instruction'], data['seconds'])))
        return cls(**dict(asdict(checked), env=dict(env), argv=expected, bon_engine=engine,
                          exec_steps=exec_steps, blend_steps=blend_steps, early_stop=early_stop,
                          action_smooth=asdict(smoothing)))
