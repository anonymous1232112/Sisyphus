from dataclasses import dataclass
import json
import os
from pathlib import Path
import warnings

import numpy as np

from bon.selector import BonSelector, load_targets
from bon.fk import YamFK


@dataclass
class Settings:
    n: int = 1
    mode: str = 'serial'
    targets: dict | None = None
    log_path: str | None = None
    fallback_reason: str | None = None
    holding_fn: object = None

    @classmethod
    def from_env(cls, env=None, mode=None):
        env = os.environ if env is None else env
        n = int(env.get('SIS_BON') or '0')
        mode = mode or env.get('SIS_BON_MODE') or 'serial'
        if mode not in ('serial', 'batch') or not 0 <= n <= 16:
            raise ValueError('SIS_BON must be 0..16; SIS_BON_MODE must be serial or batch')
        log = env.get('SIS_BON_LOG') or None
        if n <= 1:
            return cls(1, mode, log_path=log)
        try:
            target, calib = env.get('SIS_TARGET_JSON'), env.get('SIS_CALIB_JSON')
            if not target or not calib:
                raise ValueError('SIS_TARGET_JSON or SIS_CALIB_JSON is missing')
            T = np.asarray(json.loads(Path(calib).read_text())['T_base_cam'], float)
            if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError('Invalid homogeneous T_base_cam')
            targets = load_targets(target, calib)
            if targets['object'] is None and targets['region'] is None:
                raise ValueError('Neither object nor region has a valid depth target')
            if any(p is not None and (np.shape(p) != (3,) or not np.isfinite(p).all())
                   for key, p in targets.items() if key != 'table_z_base'):
                raise ValueError('Invalid target coordinates')
            return cls(n, mode, targets, log)
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
            warnings.warn('[bon] ordinary candidate-0 inference: ' + message)
            return cls(1, mode, log_path=log, fallback_reason=message)


def endpoint_spread(candidates, fk=None):
    cand = np.asarray(candidates, float)
    if cand.ndim != 3 or cand.shape[1] < 10 or cand.shape[2] != 7 or not np.isfinite(cand).all():
        raise ValueError('Expected finite (N,H>=10,7) absolute actions')
    xy = (fk or YamFK()).tip(cand[:, 9, :6])[:, :2] * 100
    distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
    std = xy.std(axis=0, ddof=0)
    return dict(endpoint_xy_cm=xy.tolist(), max_pairwise_cm=float(distance.max()),
                std_x_cm=float(std[0]), std_y_cm=float(std[1]),
                rms_radial_std_cm=float(np.linalg.norm(std)))


def repeated_batch(batch, count):
    import torch
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.ndim == 0 or value.shape[0] != 1:
                raise ValueError(f'{key}: expected one observation batch')
            out[key] = value.repeat((count,) + (1,) * (value.ndim - 1))
        elif isinstance(value, list) and len(value) == 1:
            out[key] = value * count
        else:
            out[key] = value
    return out


class SamplingFailure(RuntimeError):
    def __init__(self, message, first, noise):
        super().__init__(message)
        self.first, self.noise = first, noise


def generate(policy, batch, postprocessor, n, mode, noise=None, cancelled=lambda: False):
    import torch
    first = None
    if noise is None:
        device = next(v.device for v in batch.values() if torch.is_tensor(v))
        noise = policy.model.sample_noise((n, policy.config.chunk_size, policy.config.max_action_dim), device)
    if tuple(noise.shape) != (n, policy.config.chunk_size, policy.config.max_action_dim):
        raise ValueError('Unexpected explicit noise shape')
    try:
        if cancelled():
            return None
        if mode == 'batch':
            raw = policy.predict_action_chunk(repeated_batch(batch, n), noise=noise)
            absolute = postprocessor(raw).detach().cpu()
        elif mode == 'serial':
            chunks = []
            for index in range(n):
                if cancelled():
                    return None
                raw = policy.predict_action_chunk(batch, noise=noise[index:index+1])
                chunk = postprocessor(raw).detach().cpu()
                if index == 0:
                    first = chunk[0].numpy().copy()
                chunks.append(chunk)
            absolute = torch.cat(chunks, dim=0)
        else:
            raise ValueError('mode must be batch or serial')
        result = absolute.numpy().copy()
        if result.ndim != 3 or result.shape[0] != n or result.shape[1] < 10 or result.shape[2] != 7:
            raise ValueError(f'Invalid candidate shape {result.shape}')
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite predicted actions')
        return result
    except Exception as exc:
        raise SamplingFailure(f'{type(exc).__name__}: {exc}', first, noise[:1]) from exc


def append_log(path, record):
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('a') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
    except Exception as exc:
        warnings.warn(f'[bon] log write failed: {type(exc).__name__}: {exc}')


def make_selector(settings):
    return BonSelector(settings.targets or {}, settings.holding_fn, horizon_index=9)
