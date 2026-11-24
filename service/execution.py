import math


def validate_exec_steps(value):
    if value == 'auto':
        return value
    if not isinstance(value, str) or not value.isascii() or not value.isdigit() or not 1 <= int(value) <= 50:
        raise ValueError('SIS_BON_EXEC_STEPS must be auto or an integer in 1..50')
    return str(int(value))


def percentile90(values):
    values = sorted(float(v) for v in values)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Finite nonnegative inference latencies required')
    position = .9 * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def choose_exec_steps(setting, latencies, fps=30):
    setting = validate_exec_steps(setting)
    if setting != 'auto':
        return int(setting)
    recent = list(latencies)[-5:]
    if not recent:
        return 10
    return min(20, max(10, math.ceil(percentile90(recent) * fps) + 2))


def execution_window(latency_s, fps, candidate_steps, horizon=10):
    if (not math.isfinite(latency_s) or latency_s < 0 or
            not math.isfinite(fps) or fps <= 0 or type(horizon) is not int or not 1 <= horizon <= 50):
        raise ValueError('Invalid inference latency / FPS / execution steps')
    start = math.ceil(latency_s * fps)
    end = start + horizon
    if end > min(50, candidate_steps):
        raise RuntimeError(f'BoN result expired: d={start}, exec={horizon}, available={candidate_steps}')
    return start, end
