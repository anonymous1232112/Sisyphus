from functools import wraps
import math
import os


def install_pi05_noise_scale(pi05_module):
    try:
        scale = float(os.environ.get("PI05_NOISE_SCALE", "1.0"))
    except ValueError:
        raise ValueError("PI05_NOISE_SCALE must be a finite nonnegative number") from None
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("PI05_NOISE_SCALE must be a finite nonnegative number")

    original = getattr(pi05_module.sample_noise, "_yam_original_sampler", pi05_module.sample_noise)
    if scale == 1.0:
        pi05_module.sample_noise = original
    else:
        first_sample = True

        @wraps(original)
        def scaled_noise(shape, device):
            nonlocal first_sample
            noise = original(shape, device)
            result = noise * scale
            if first_sample:
                print(f"[lowmem] initial-noise sampler active, scale={scale}", flush=True)
                first_sample = False
            return result

        scaled_noise._yam_original_sampler = original
        pi05_module.sample_noise = scaled_noise
    print(f"[lowmem] noise scale {scale}", flush=True)
    return scale
