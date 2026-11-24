import argparse
import os
from pathlib import Path
import select
import signal
import socket
import sys
import threading
import time

from policy.common import ROOT, Task, address, packet, receive


class TaskCancelled(Exception):
    pass


class Control:
    def __init__(self):
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.events = dict(exit_early=False, rerecord_episode=False, stop_recording=False)
        self.reason = ''
        self.code = 0
        self.cancel_time = None
        self.lock = threading.Lock()

    def cancel(self, reason, code=130):
        with self.lock:
            if self.cancel_time is None:
                self.reason, self.code, self.cancel_time = reason, code, time.monotonic()
            self.events.update(exit_early=True, stop_recording=True)
            self.cancelled.set()

    def check(self):
        if self.cancelled.is_set():
            raise TaskCancelled(self.reason)


class TerminalKeys:
    def __init__(self, dispatch):
        self.dispatch, self.old, self.fd = dispatch, None, None
        self.stopped = threading.Event()

    def start(self):
        import termios
        import tty
        if not sys.stdin.isatty():
            raise RuntimeError('Real policy server MUST run in its own interactive terminal')
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped.is_set():
            if not select.select([self.fd], [], [], .05)[0]:
                continue
            char = os.read(self.fd, 1)
            if char == b'\x1b':
                if select.select([self.fd], [], [], .03)[0]:
                    os.read(self.fd, 1)
                    continue
                self.dispatch('esc')
            elif char.lower() in (b'q', b'n'):
                self.dispatch(char.decode().lower())

    def stop(self):
        import termios
        self.stopped.set()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)
        if self.old is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
            self.old = None


class Service:
    def __init__(self, model, backend, cleanup_grace=30, hard_exit=os._exit):
        self.model, self.backend = str(Path(model).resolve()), backend
        self.cleanup_grace, self.hard_exit = cleanup_grace, hard_exit
        self.busy = threading.Lock()
        self.stopping, self.ready = threading.Event(), threading.Event()
        self.active = None
        self.fatal = False

    def key(self, key):
        control = self.active
        if control is not None:
            if key in ('esc', 'q'):
                control.cancel('Esc in policy-server terminal', 130)
                print('\n[policy-server] ESC: cancelling task; no automatic return.', flush=True)
            elif key == 'n':
                control.events['exit_early'] = True

    def shutdown(self, *args):
        self.stopping.set()
        if self.active is not None:
            self.active.cancel('Policy service shutting down', 130)

    def handle(self, conn):
        control = None
        acquired = False
        send_lock = threading.Lock()
        terminal_sent = False

        def finish(code, **info):
            nonlocal terminal_sent
            with send_lock:
                if terminal_sent:
                    return
                terminal_sent = True
                try:
                    packet(conn, dict(event='done', exit_code=code, **info))
                except OSError:
                    pass

        try:
            conn.settimeout(5)
            with conn.makefile('rb') as stream:
                task = Task.parse(receive(stream))
            if task.model != self.model:
                finish(64, error='MODEL does not match the resident model; nothing was connected')
                return
            if task.root.exists() or task.root.is_symlink():
                finish(2, error='Dataset already exists; nothing was changed')
                return
            acquired = self.busy.acquire(blocking=False)
            if not acquired:
                finish(75, error='Service is busy; task was NOT queued')
                return
            if self.stopping.is_set():
                finish(69, error='Service is stopping')
                return
            control = Control()
            self.active = control
            packet(conn, dict(event='accepted', timeout_s=task.timeout_s))
            start = time.monotonic()

            def watchdog():
                while not control.done.wait(.05):
                    now = time.monotonic()
                    if now - start >= task.timeout_s:
                        control.cancel('Task deadline exceeded', 124)
                    try:
                        if select.select([conn], [], [], 0)[0] and not conn.recv(1, socket.MSG_PEEK):
                            control.cancel('Client disconnected', 130)
                    except OSError:
                        control.cancel('Client disconnected', 130)
                    if control.cancel_time is not None and now - control.cancel_time >= self.cleanup_grace:
                        self.stopping.set()
                        finish(control.code or 124, error='Cleanup deadline exceeded; policy service exiting',
                               fatal=True, cleanup_complete=False)
                        print('[policy-server] HARD TIMEOUT: terminating ONLY this policy-service process.', flush=True)
                        self.hard_exit(control.code or 124)
                        return

            threading.Thread(target=watchdog, daemon=True).start()
            print(f'[policy-server] TASK_BEGIN dataset={task.dataset} seed={task.seed}', flush=True)
            try:
                result = self.backend.run(task, control)
                finish(control.code, **(result or {}), error=control.reason or None,
                       cleanup_complete=True, elapsed_s=round(time.monotonic()-start, 3))
            except Exception as exc:
                self.stopping.set()
                self.fatal = True
                finish(1, error=f'{type(exc).__name__}: {exc}', fatal=True, cleanup_complete=False)
                print(f'[policy-server] FATAL {type(exc).__name__}: {exc}', flush=True)
            finally:
                control.done.set()
                self.active = None
                print('[policy-server] TASK_END', flush=True)
        except Exception as exc:
            finish(64, error=f'{type(exc).__name__}: {exc}')
        finally:
            if control is not None:
                control.done.set()
                if self.active is control:
                    self.active = None
            if acquired:
                self.busy.release()
            conn.close()

    def serve(self, listener):
        listener.settimeout(.2)
        self.ready.set()
        print('[policy-server] READY ' + str(listener.getsockname()) +
              '; model resident, cameras released; Esc/q cancels current task, Ctrl+C exits service.', flush=True)
        try:
            while not self.stopping.is_set():
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self.handle, args=(conn,), daemon=True).start()
        finally:
            listener.close()
            control = self.active
            if control is not None:
                control.cancel('Service shutting down', 130)
                if not control.done.wait(self.cleanup_grace + 1):
                    self.hard_exit(124)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--listen', default='tcp://127.0.0.1:8767')
    a = ap.parse_args()
    os.chdir(ROOT)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    from policy.backend import ResidentBackend
    backend = ResidentBackend(a.model)
    service = Service(a.model, backend)
    keys = TerminalKeys(service.key)
    keys.start()
    signal.signal(signal.SIGINT, service.shutdown)
    signal.signal(signal.SIGTERM, service.shutdown)
    try:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(address(a.listen))
            backend.initialize()
            listener.listen(8)
            service.serve(listener)
    finally:
        keys.stop()
    if service.fatal:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)


if __name__ == '__main__':
    raise SystemExit(main())
