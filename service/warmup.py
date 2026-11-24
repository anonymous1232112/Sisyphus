import time
import numpy as np

from .sampling import TimedSampler


def fake_observation(config):
    obs = {'observation.state': np.zeros(7, dtype=np.float32)}
    obs['observation.state'][6] = 1.
    for name, feature in config.input_features.items():
        if name.startswith('observation.images.'):
            channels, height, width = feature.shape
            obs[name] = np.zeros((height, width, channels), dtype=np.uint8)
    if not any(k.startswith('observation.images.') for k in obs):
        raise ValueError('Warmup requires configured image features')
    return obs


def warmup_model(policy, model_path):
    import torch
    from lerobot.policies import make_pre_post_processors
    device = next(policy.parameters()).device
    previous = (policy.config.rtc_config, getattr(policy, 'rtc_processor', None),
                getattr(policy.model, 'rtc_processor', None))
    started = time.perf_counter()
    try:
        policy.config.rtc_config = None
        policy.rtc_processor = policy.model.rtc_processor = None
        policy.reset()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=model_path,
            preprocessor_overrides={'device_processor': {'device': str(device)},
                                    'rename_observations_processor': {'rename_map': {}}})
        gen = torch.Generator(device=device).manual_seed(0)
        noise = torch.randn((5, policy.config.chunk_size, policy.config.max_action_dim),
                            device=device, generator=gen) * .5
        sample = TimedSampler(policy, pre, post, device=str(device), n=5,
            task='pick up the red cube and place it at the center of the red region',
            robot_type='yam_follower')(fake_observation(policy.config), noise=noise)
        if sample.candidates is None or sample.candidates.shape != (5, 50, 7):
            raise RuntimeError('N=5 warmup did not produce a complete candidate batch')
        return dict(n=5, sigma=.5, fake_observation=True, seconds=time.perf_counter()-started,
                    timing_s=sample.timing)
    finally:
        policy.reset()
        policy.config.rtc_config, policy.rtc_processor, policy.model.rtc_processor = previous
