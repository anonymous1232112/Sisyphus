from dataclasses import asdict
import json
import math
import os
import re
import socket
import sys

from policy.common import ENV_KEYS, ROOT, address, packet, receive

SERVER = 'tcp://127.0.0.1:8770'
ACCEPT_TIMEOUT = 30.
MARGIN = 150.
USAGE = 'Usage: python -m service.client MODEL DATASET "INSTRUCTION" SEED [SECONDS]'


class UsageError(ValueError):
    pass


def positional(argv):
    if not 4 <= len(argv) <= 5:
        raise UsageError('expected 4 or 5 positional arguments')
    model, dataset, instruction, seed = argv[:4]
    seconds = argv[4] if len(argv) == 5 else '40'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', dataset) or not dataset.startswith('rollout_'):
        raise UsageError('Dataset name must start with rollout_ and use [A-Za-z0-9_.-].')
    if not instruction.strip():
        raise UsageError('Instruction must not be empty.')
    if not re.fullmatch(r'[0-9]{1,10}', seed) or int(seed) > 4294967295:
        raise UsageError('Seed must be an integer in [0, 4294967295].')
    if not re.fullmatch(r'[0-9]+(?:[.][0-9]+)?', seconds) or not float(seconds) > 0:
        raise UsageError('Duration must be positive seconds.')
    return os.path.realpath(os.path.expanduser(model)), dataset, instruction, int(seed), seconds


def timeout(seconds, env):
    raw = env.get('SIS_TASK_TIMEOUT_S')
    value = float(raw) if raw else min(3600., float(seconds) + 600.)
    if not math.isfinite(value) or not float(seconds) < value <= 3600:
        raise UsageError('SIS_TASK_TIMEOUT_S must exceed SECONDS and be at most 3600')
    return value


def build(model, dataset, instruction, seed, seconds, env):
    from service.resident import resident_args
    from service.smoothing import SmoothSettings
    from service.holding import H_RULE_VERSION
    from service.protocol import settings_from_env
    flags = {key: env.get(key, default) for key, default in
             (('SIS_A2_ABORT_ACTION_ENABLED', '1'), ('SIS_RELOCATE_ON_EMPTY', '0'), ('SIS_CONTINUATION', '0'))}
    if any(value not in ('0', '1') for value in flags.values()):
        raise UsageError('SIS_A2_ABORT_ACTION_ENABLED, SIS_RELOCATE_ON_EMPTY and SIS_CONTINUATION must be 0 or 1')
    try:
        abort = settings_from_env(env)
    except ValueError as exc:
        raise UsageError(f'Invalid abort settings: {exc}') from exc
    return dict(
        version=1, model=model, dataset=dataset, instruction=instruction, seed=seed,
        seconds=seconds, env={key: env.get(key) for key in ENV_KEYS},
        argv=resident_args(model, dataset, instruction, seconds),
        timeout_s=timeout(seconds, env),
        bon_engine=env.get('SIS_BON_ENGINE') or 'async_shared',
        exec_steps=env.get('SIS_BON_EXEC_STEPS') or 'auto',
        blend_steps=env.get('SIS_BON_BLEND_STEPS') or '5',
        early_stop=env.get('SIS_EARLY_STOP') or '1',
        action_smooth=asdict(SmoothSettings.from_env(env)),
        abort_settings=abort, abort_action_enabled=flags['SIS_A2_ABORT_ACTION_ENABLED'] == '1',
        relocate_on_empty=flags['SIS_RELOCATE_ON_EMPTY'] == '1', continuation=flags['SIS_CONTINUATION'] == '1',
        h_rule_version=H_RULE_VERSION)


def send(obj, server):
    try:
        sock = socket.create_connection(address(server), timeout=5.)
    except (OSError, ValueError) as exc:
        print(f'[client] service not reachable at {server}: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 69
    try:
        with sock:
            packet(sock, obj)
            sock.settimeout(ACCEPT_TIMEOUT)
            with sock.makefile('rb') as stream:
                reply = receive(stream)
                if reply.get('event') == 'accepted':
                    print(f'[client] accepted {obj["dataset"]}', flush=True)
                    sock.settimeout(float(obj['timeout_s']) + MARGIN)
                    reply = receive(stream)
                if reply.get('event') != 'done':
                    print('[client] unexpected packet: ' + json.dumps(reply, ensure_ascii=False), file=sys.stderr)
                    return 76
                print('[client] DONE ' + json.dumps(reply, ensure_ascii=False), flush=True)
                code = reply.get('exit_code')
                return code if type(code) is int and 0 <= code <= 255 else 76
    except socket.timeout:
        print('[client] no terminal packet before the deadline', file=sys.stderr)
        return 124
    except KeyboardInterrupt:
        print('\n[client] interrupted: the service cancels and saves', file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f'[client] connection error: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 76


def main(argv=None):
    from service.protocol import AbortTask
    env = os.environ
    try:
        model, dataset, instruction, seed, seconds = positional(sys.argv[1:] if argv is None else argv)
        if (ROOT / 'data' / dataset).exists():
            raise UsageError(f'Dataset already exists: {ROOT / "data" / dataset}')
        obj = build(model, dataset, instruction, seed, seconds, env)
        try:
            AbortTask.parse(json.loads(json.dumps(obj)))
        except (ValueError, KeyError, TypeError) as exc:
            raise UsageError(f'Task rejected: {type(exc).__name__}: {exc}') from exc
    except UsageError as exc:
        print(str(exc), file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2
    return send(obj, env.get('SIS_POLICY_SERVER') or SERVER)


if __name__ == '__main__':
    raise SystemExit(main())
