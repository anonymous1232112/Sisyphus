from unittest.mock import patch
import sys


def main():
    import policy.backend as yam_policy_backend
    import policy.server as yam_policy_server
    from service.backend import SharedResidentBackend, AtomicControl, ResidentService
    from service.protocol import AbortTask as ResidentTask
    if not any(a == '--listen' or a.startswith('--listen=') for a in sys.argv[1:]):
        sys.argv += ['--listen', 'tcp://127.0.0.1:8770']
    with patch.object(yam_policy_backend, 'ResidentBackend', SharedResidentBackend), \
            patch.object(yam_policy_server, 'Task', ResidentTask), \
            patch.object(yam_policy_server, 'Control', AtomicControl), \
            patch.object(yam_policy_server, 'Service', ResidentService):
        return yam_policy_server.main()


if __name__ == '__main__':
    raise SystemExit(main())
